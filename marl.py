import torch
from trl import GRPOConfig, GRPOTrainer


# class MultiagentTrainer:
#   def __init__(self, config: GRPOConfig):
#     pass

#   def train(self, resume_from_checkpoint):
#     # TODO: resuming

#     for step in range(self.state.global_step, self.args.max_step):
#       loss_stats = self.training_step()
#       # TODO: update progress bar
#       # TODO: logging

#   def training_step(self):
#     # TODO: everything

#     with trainer.accelerator:
#       pass


# grpo.compute_loss
# called by Trainer.training_step
# called by Trainer.inner_training_loop
# called by Trainer.train


def _single_agent_training_step(agent, batch, experiment):
  model = agent['model']
  trainer = agent['trainer']
  model.train()
  if hasattr(trainer.optimizer, 'train') and callable(trainer.optimizer.train):
    trainer.optimizer.train()

  # gradient accumulation and AMP
  with trainer.accelerator.accumulate(model):
    # normally, self._prepare_inputs is called here
    # which, in GRPO, calls _generate_and_score_completions
    # unfortunately, that is incompatible with a multi-agent setup: we need the completions from all agents here.
    # and this is where the fun starts
    prompts = batch['prompt']
    with torch.no_grad():
      completions_ids = trainer.generate(
        prompts,
        max_new_tokens=trainer.args.max_completion_length,
        temperature=trainer.args.temperature,
        top_p=trainer.args.top_p,
      )
    completion_texts = trainer.tokenizer.batch_decode(
      completions_ids, skip_special_tokens=True
    )
    rewards = experiment.get_single_agent_rewards(prompts, completion_texts)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)

    # --- Phase 2: Assembling Inputs for Loss Calculation ---
    # We now manually assemble the dictionary that `trainer.compute_loss` expects.
    # This involves calculating advantages and getting log probabilities.

    # Calculate advantages from rewards
    num_prompts = len(prompts)
    num_generations = trainer.args.num_generations
    grouped_rewards = rewards_tensor.view(num_prompts, num_generations)
    advantages = grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)
    if trainer.args.scale_rewards:
      advantages /= grouped_rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = advantages.view(-1)

    # Tokenize and get log probabilities
    tokenized_prompts = trainer.tokenizer(
      prompts, padding=True, truncation=True, return_tensors='pt'
    ).to(model.device)
    tokenized_completions = trainer.tokenizer(
      completion_texts, padding=True, truncation=True, return_tensors='pt'
    ).to(model.device)

    full_tokenized_ids = torch.cat(
      (
        tokenized_prompts.input_ids.repeat_interleave(num_generations, dim=0),
        tokenized_completions.input_ids,
      ),
      dim=1,
    )
    full_attention_mask = torch.cat(
      (
        tokenized_prompts.attention_mask.repeat_interleave(num_generations, dim=0),
        tokenized_completions.attention_mask,
      ),
      dim=1,
    )

    with torch.no_grad():
      logprobs_data = trainer._get_per_token_logps_and_entropies(
        model,
        full_tokenized_ids,
        full_attention_mask,
        logits_to_keep=tokenized_completions.input_ids.shape[1],
      )

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
      'old_per_token_logps': logprobs_data['logps'].detach(),
      'ref_per_token_logps': None,  # beta=0, so no ref model needed
    }

    # --- Phase 3: Loss Computation and Optimization ---

    # This is equivalent to `loss = self.compute_loss(model, inputs)` in the original Trainer
    loss = trainer.compute_loss(model, loss_inputs)

    # This is equivalent to `self.accelerator.backward(loss)`.
    # It correctly handles gradient scaling for AMP and gradient averaging for DDP.
    # The `loss.mean()` for multi-GPU is NOT needed.
    trainer.accelerator.backward(loss)

    # These calls are performed by the `accelerator.accumulate` context manager
    # only when the correct number of accumulation steps has been reached.
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()

  # --- Phase 4: Memory Management ---
  # This is good practice.
  detached_loss = loss.detach().clone()
  del loss, loss_inputs, logprobs_data, advantages, rewards_tensor
  # if torch.cuda.is_available():
  #     torch.cuda.empty_cache()

  return detached_loss
