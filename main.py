import os
import torch
os.environ["HF_HOME"] = os.path.abspath("./.hf_cache")
from peft import LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback, ProgressCallback
from transformers.trainer_utils import get_last_checkpoint
from util import get_current_commit, lora_print_trainable_parameters
from number_game import NumberGameExperiment
import wandb

def load_model(lora_rank):
  device = 'auto'
  model_id = 'Qwen/Qwen3-8B'

  # qwen has endoftext as pad token
  tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto',
    device_map=device,
  )

  if lora_rank:
    lora_config = LoraConfig(
      r=lora_rank,
      lora_alpha=2*lora_rank,
      lora_dropout=0.05,
      task_type=TaskType.CAUSAL_LM,
      target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
      ],
    )
    model = get_peft_model(model, lora_config)

  # 8B: rank ~ trainable%:
  # 256 ~ 8%, 1024 ~ 25%, 2048 ~ 40%
  lora_print_trainable_parameters(model)
  return model, tokenizer


# i hate it
def on_log(self, args, state, control, logs=None, **kwargs):
  if state.is_local_process_zero and self.training_bar is not None:
    _ = logs.pop("total_flos", None)
ProgressCallback.on_log = on_log


class CustomCheckpointCallback(TrainerCallback):
  def __init__(self):
    super().__init__()
    self.commit_hash = get_current_commit()

  def on_save(self, args, state, control, **kwargs):
    if state.is_world_process_zero and self.commit_hash:
      checkpoint_step = state.global_step
      checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
      print('saving to:', checkpoint_dir)
      with open(os.path.join(checkpoint_dir, "git-hash"), 'w') as f:
        f.write(self.commit_hash)

def should_resume(checkpoint_path, override=False):
  if override:
    return True
  last_checkpoint_dir = get_last_checkpoint(checkpoint_path)
  if not last_checkpoint_dir:
    return False
  try:
    with open(os.path.join(last_checkpoint_dir, "git-hash"), 'r') as f:
      checkpoint_commit = f.read().strip()
  except Exception:
    return False
  resuming = checkpoint_commit == get_current_commit()
  print(f"resuming from {last_checkpoint_dir}" if resuming else "not resuming")
  return resuming

def main():
  os.environ['WANDB_PROJECT'] = 'llm-comm-opt'
  os.environ['WANDB_DIR'] = os.path.abspath("./.wandb")
  os.environ['WANDB_ARTIFACT_DIR'] = os.path.abspath("./.wandb_artifacts")
  os.environ['WANDB_CACHE_DIR'] = os.path.abspath("./.wandb_cache")
  os.environ['WANDB_DATA_DIR'] = os.path.abspath("./.wandb_data")

  lora_rank = 2048

  wandb.init(config={
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "lora_rank": lora_rank,
  })

  checkpoint_path = "checkpoints"
  model, tokenizer = load_model(lora_rank)

  # for grpo, scale gen count not batch sz
  generation_count = 12
  # 32-64 for 4xh100
  if torch.cuda.device_count() == 4:
    generation_count = 48
  # 64-128 for 8xh100
  if torch.cuda.device_count() == 8:
    generation_count = 96

  grpo_config = GRPOConfig(
    # KL to reference model
    beta=0,
    output_dir=checkpoint_path,
    num_generations=generation_count,
    report_to="wandb",
    # log freq
    logging_steps=1,
    log_completions=True,
    wandb_log_unique_prompts=True,
    max_steps=1_000,
    save_steps=100,
    # needs to be divisible by num_generations
    per_device_train_batch_size=generation_count,
    # question-level difficulty bias
    scale_rewards=False,
    # seems to be getting forcibly reset to 0.95 with qwen3
    top_p=1.0,
    temperature=1.5,
  )

  experiment = NumberGameExperiment()

  grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=experiment.mk_dataset(5_000, tokenizer),
    reward_funcs=experiment.mk_reward_func(model, tokenizer),
    args=grpo_config,
    callbacks=[CustomCheckpointCallback()]
  )
  grpo_trainer.train(resume_from_checkpoint=should_resume(checkpoint_path, override=False))


if __name__ == '__main__':
  main()
