import os
import random
import torch
from tqdm.auto import tqdm

# --- User-provided setup ---
# from util import select_least_mem_used_gpus
# select_least_mem_used_gpus(6)
os.environ['HF_HOME'] = os.path.abspath('./.hf_cache')
# --- End user-provided setup ---

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  TrainerCallback,
  TrainerState,
  TrainerControl,
)
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR
from datasets import Dataset
import wandb


class MultiAgentTrainer:
  """
  A custom trainer to handle the simultaneous training of multiple agents
  using Independent Learning (IL) with a shared reward.

  This trainer orchestrates the multi-agent interaction but uses individual
  TRL GRPOTrainers as backends to handle the complex optimization logic for each agent.
  It implements its own training loop and does not use `trainer.train()`.
  """

  def __init__(self, agents: list, experiment, args: GRPOConfig, **kwargs):
    """
    Initializes the MultiAgentTrainer.

    Args:
        agents (list): A list of dictionaries, where each dict contains a 'model' and 'tokenizer'.
        experiment: An object that defines the task, providing methods to create datasets
                    and calculate rewards.
        args (GRPOConfig): The configuration for training.
    """
    self.agents = agents
    self.num_agents = len(agents)
    self.experiment = experiment
    self.args = args
    self.device = args.device

    # --- Initialize a separate GRPOTrainer for each agent ---
    for i, agent in enumerate(self.agents):
      # We need a dummy reward function to satisfy the trainer's __init__
      dummy_reward_func = lambda completions, **kwargs: [0.0] * len(completions)

      trainer = GRPOTrainer(
        model=agent['model'],
        args=self.args,
        tokenizer=agent['tokenizer'],
        reward_funcs=dummy_reward_func,
        callbacks=kwargs.get('callbacks', []),
      )
      # The GRPOTrainer creates its own optimizer and scheduler. We need them for the manual loop.
      trainer.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
      agent['trainer'] = trainer
      agent['model'].to(self.device)

    # --- Mimic Hugging Face Trainer's state and control flow ---
    self.state = TrainerState(
      is_local_process_zero=self.args.is_local_process_zero,
      is_world_process_zero=self.args.is_world_process_zero,
    )
    self.control = TrainerControl()
    self.is_in_train = False
    self.train_dataloader = None

    # This is important for logging
    if self.args.report_to == 'wandb':
      wandb.watch(
        models=[agent['model'] for agent in self.agents],
        log='all',
        log_freq=args.logging_steps,
      )

  def train(self, resume_from_checkpoint: Union[bool, str, None] = None):
    """
    Main training entry point. This custom loop replaces the default `trainer.train()`.
    """
    args = self.args
    self.is_in_train = True

    # --- TODO: Implement Checkpoint Loading if needed ---
    # if resume_from_checkpoint:
    #     self._load_from_checkpoint(resume_from_checkpoint)

    # --- Dataloader Setup ---
    # We create one dataloader for the prompts/context of the game
    # The dataset size here is just for show, as we generate prompts on the fly.
    # It's mainly used to determine the number of steps.
    train_dataset = self.experiment.mk_dataset(
      self.args.max_steps * self.args.per_device_train_batch_size,
      self.agents[0]['tokenizer'],
    )
    self.train_dataloader = self.agents[0]['trainer'].get_train_dataloader()

    print('***** Running Multi-Agent Training *****')
    print(f'  Num agents = {self.num_agents}')
    print(f'  Max steps = {args.max_steps}')
    print(f'  Batch size per device = {args.per_device_train_batch_size}')
    print(f'  Num generations per prompt = {args.num_generations}')
    print(
      f'  Total completions per step = {args.per_device_train_batch_size * args.num_generations}'
    )

    progress_bar = tqdm(
      range(args.max_steps),
      desc='Training Steps',
      disable=not self.args.is_local_process_zero,
    )

    for step in range(self.state.global_step, args.max_steps):
      # --- The Core Multi-Agent Step ---
      loss_stats = self._training_step()

      self.state.global_step += 1
      progress_bar.update(1)

      # --- Logging ---
      if (
        self.args.is_local_process_zero
        and self.state.global_step % args.logging_steps == 0
      ):
        logs = {f'loss/agent_{i}': loss for i, loss in enumerate(loss_stats)}
        logs['step'] = self.state.global_step
        if self.args.report_to == 'wandb':
          wandb.log(logs)
        else:
          print(logs)

      # --- TODO: Implement Saving if needed ---
      # if self.state.global_step % args.save_steps == 0:
      #     self._save_checkpoint()

      if self.control.should_training_stop:
        break

    print('\n\nTraining completed.\n\n')
    self.is_in_train = False

  def _training_step(self):
    """
    Performs one step of the simultaneous multi-agent training loop.
    This is a manual implementation of the logic within `GRPOTrainer.train()`.
    """
    # --- Phase A: Rollout ---
    batch_size = self.args.per_device_train_batch_size
    prompts, contexts = self.experiment.get_prompt_batch(batch_size)

    agent_rollout_data = [
      {'prompts': [], 'completions': []} for _ in range(self.num_agents)
    ]

    for i, agent_data in enumerate(agent_rollout_data):
      agent_prompts = [p[i] for p in prompts]
      agent_data['prompts'] = agent_prompts

      # Use the agent's dedicated trainer to generate text
      with torch.no_grad():
        agent_data['completions'] = self.agents[i]['trainer'].generate(
          agent_prompts,
          max_new_tokens=self.args.max_completion_length,
          temperature=self.args.temperature,
          top_p=self.args.top_p,
        )

    # --- Phase B: Shared Reward Calculation ---
    # The reward function needs all completions to calculate the team reward
    all_rewards = self.experiment.get_reward_batch(
      prompts=prompts,
      completions=[agent_data['completions'] for agent_data in agent_rollout_data],
      contexts=contexts,
    )
    # Convert rewards to a tensor for processing
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)

    # --- Phase C: Optimization for each agent ---
    all_losses = []
    for i, agent_data in enumerate(agent_rollout_data):
      trainer = self.agents[i]['trainer']
      model = self.agents[i]['model']

      # This is the crucial part: we manually replicate the logic of `_prepare_inputs`
      # and `compute_loss` from the GRPOTrainer source.

      # 1. Prepare inputs for the loss function
      # This involves tokenizing, calculating logprobs, and advantages.
      # We can borrow the internal methods from the agent's own trainer.

      # We need to group the data by prompt for advantage calculation
      num_prompts = len(agent_data['prompts'])
      num_generations = self.args.num_generations

      # Calculate advantages (relative rewards)
      grouped_rewards = rewards_tensor.view(num_prompts, num_generations)
      mean_rewards = grouped_rewards.mean(dim=1, keepdim=True)
      std_rewards = grouped_rewards.std(dim=1, keepdim=True)

      advantages = grouped_rewards - mean_rewards
      if self.args.scale_rewards:
        advantages /= std_rewards + 1e-8

      # Flatten advantages to match the shape of completions
      advantages = advantages.view(-1)

      # Tokenize prompts and completions to get logprobs
      tokenized_prompts = trainer.tokenizer(
        agent_data['prompts'], padding=True, truncation=True, return_tensors='pt'
      ).to(self.device)
      tokenized_completions = trainer.tokenizer(
        agent_data['completions'], padding=True, truncation=True, return_tensors='pt'
      ).to(self.device)

      full_tokenized = {
        'input_ids': torch.cat(
          (
            tokenized_prompts.input_ids.repeat_interleave(num_generations, dim=0),
            tokenized_completions.input_ids,
          ),
          dim=1,
        ),
        'attention_mask': torch.cat(
          (
            tokenized_prompts.attention_mask.repeat_interleave(num_generations, dim=0),
            tokenized_completions.attention_mask,
          ),
          dim=1,
        ),
      }

      with torch.no_grad():
        logprobs_data = trainer._get_per_token_logps_and_entropies(
          model,
          full_tokenized['input_ids'],
          full_tokenized['attention_mask'],
          logits_to_keep=tokenized_completions.input_ids.shape[1],
        )

      # 2. Construct the batch dictionary that `compute_loss` expects
      loss_inputs = {
        'prompt_ids': tokenized_prompts.input_ids.repeat_interleave(
          num_generations, dim=0
        ),
        'prompt_mask': tokenized_prompts.attention_mask.repeat_interleave(
          num_generations, dim=0
        ),
        'completion_ids': tokenized_completions.input_ids,
        'completion_mask': tokenized_completions.attention_mask,
        'advantages': advantages,
        'old_per_token_logps': logprobs_data[
          'logps'
        ].detach(),  # Use current logprobs as old ones for first iteration
        'ref_per_token_logps': None,  # Since beta=0
      }

      # 3. Compute loss
      # We need to wrap this in the model's training context
      with trainer.accelerator.accumulate(model):
        loss = trainer.compute_loss(model, loss_inputs)
        all_losses.append(loss.detach().item())

        # 4. Backward pass and optimizer step
        trainer.accelerator.backward(loss)
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.optimizer.zero_grad()

    return all_losses

  def _save_checkpoint(self):
    # TODO: Implement this if needed, following the logic in the previous version
    pass

  def _load_from_checkpoint(self, resume_from_checkpoint):
    # TODO: Implement this if needed, following the logic in the previous version
    pass


# --- Example Experiment Definition ---
class CollaborativeSummarizationExperiment:
  def __init__(self, n_agents=2):
    self.n_agents = n_agents
    # In a real scenario, you'd load a summarization dataset
    self.documents = [
      "The James Webb Space Telescope, a marvel of modern engineering, was launched on December 25, 2021. Its primary mission is to succeed the Hubble Space Telescope as NASA's flagship astrophysics mission. The telescope is designed to see the early universe, observing the formation of the first galaxies and stars.",
      'Jupiter, the largest planet in our solar system, is a gas giant primarily composed of hydrogen and helium. It features the Great Red Spot, a gigantic storm larger than Earth that has been raging for centuries. Several missions, including Juno and Galileo, have studied its complex atmosphere and powerful magnetic field.',
    ] * 100  # Repeat for more data

  def get_prompt_batch(self, batch_size):
    """Gets a batch of prompts and the original contexts."""
    batch_docs = random.sample(self.documents, batch_size)
    prompts = []
    for doc in batch_docs:
      words = doc.split()
      len_words = len(words)
      chunk1 = ' '.join(words[: int(len_words * 0.6)])
      chunk2 = ' '.join(words[int(len_words * 0.4) :])

      agent0_prompt = (
        f"You are Agent 0. Summarize your part of the document concisely: '{chunk1}'"
      )
      agent1_prompt = (
        f"You are Agent 1. Summarize your part of the document concisely: '{chunk2}'"
      )
      prompts.append([agent0_prompt, agent1_prompt])

    return prompts, batch_docs

  def get_reward_batch(self, prompts, completions, contexts):
    """
    Calculates the shared team reward for a batch of games.
    `completions` is a list of lists: [[agent0_comps], [agent1_comps]]
    """
    batch_size = len(prompts)
    num_generations = len(completions[0]) // batch_size

    rewards = []
    for i in range(batch_size * num_generations):
      agent_summaries = [agent_comps[i] for agent_comps in completions]
      full_summary = ' '.join(agent_summaries)

      # --- Dummy Reward Logic ---
      similarity_score = (
        10 if 'Telescope' in full_summary or 'Jupiter' in full_summary else 0
      )
      length_penalty = 0.05 * len(full_summary)

      summary1_words = set(agent_summaries[0].split())
      summary2_words = set(agent_summaries[1].split())
      overlap = len(summary1_words.intersection(summary2_words))
      redundancy_penalty = 0.5 * overlap

      reward = similarity_score - length_penalty - redundancy_penalty
      rewards.append(reward)

    return rewards

  def mk_dataset(self, num_samples, tokenizer):
    # This is just to satisfy the dataloader, the prompts are generated on the fly
    return Dataset.from_dict({'prompt': [''] * num_samples})


def load_model(lora_rank):
  device = 'auto'
  model_id = 'Qwen/Qwen3-0.6B'  # Use a smaller model for faster iteration

  tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto',
    device_map=device,
  )

  if lora_rank:
    lora_config = LoraConfig(
      r=lora_rank,
      lora_alpha=2 * lora_rank,
      lora_dropout=0.05,
      task_type=TaskType.CAUSAL_LM,
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)

  lora_print_trainable_parameters(model)
  return model, tokenizer


def main():
  # --- Setup ---
  os.environ['WANDB_PROJECT'] = 'llm-marl-comm-opt'
  wandb.init(config={'slurm_job_id': os.environ.get('SLURM_JOB_ID')})

  N_AGENTS = 2
  LORA_RANK = 16

  # --- Create Agents ---
  agents = []
  for i in range(N_AGENTS):
    print(f'Loading model for agent {i}...')
    model, tokenizer = load_model(LORA_RANK)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    agents.append({'model': model, 'tokenizer': tokenizer})

  # --- Config ---
  grpo_config = GRPOConfig(
    beta=0.0,
    output_dir='marl_checkpoints',
    num_generations=4,  # Keep low for CPU testing
    report_to='wandb',
    logging_steps=1,
    max_steps=100,  # Run for fewer steps to test the loop
    save_steps=50,
    per_device_train_batch_size=2,  # 2 prompts per device
    scale_rewards=False,
    temperature=1.0,
  )

  # --- Experiment and Trainer ---
  experiment = CollaborativeSummarizationExperiment(n_agents=N_AGENTS)

  multi_agent_trainer = MultiAgentTrainer(
    agents=agents,
    experiment=experiment,
    args=grpo_config,
  )

  # --- Train ---
  multi_agent_trainer.train()


if __name__ == '__main__':
  main()
