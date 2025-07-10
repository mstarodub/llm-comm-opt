import os
import random
import torch
os.environ["HF_HOME"] = os.path.abspath("./.hf_cache")
from peft import LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from util import get_current_commit, lora_print_trainable_parameters
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

def mk_prompt(sys_prompt, prompt):
  return [
    {
      'role': 'system',
      'content': sys_prompt,
    },
    {'role': 'user', 'content': prompt},
  ]

def mk_input(model, tokenizer, sys_prompt, prompt):
  full_prompt = mk_prompt(sys_prompt, prompt)
  chat = tokenizer.apply_chat_template(
    full_prompt,
    tokenize=False,
    # turn this off and do continue_final_message=True for continuing an assistant message
    add_generation_prompt=True,
    enable_thinking=False,
  )
  tokenized = tokenizer([chat], padding=True, return_tensors='pt').to(model.device)
  return tokenized


def inference(model, tokenizer, input):
  max_new_tokens = 2048
  # TODO: do_sample=True ?
  generated_ids = model.generate(**input, max_new_tokens=max_new_tokens)
  # select first and only batch; select everything after input tokens (this has input + output by default)
  output_ids = generated_ids[0][len(input.input_ids[0]) :]

  content = tokenizer.decode(output_ids, skip_special_tokens=False)
  return content.strip()


def parse_recv_msg(msg):
  num_s = ''.join(ch for ch in msg if ch.isdigit())
  try:
    res = int(num_s)
  except ValueError:
    res = None
  return res

# this custom reward function will be called with whatever columns are in the dataset as kwargs
def rewards(model, tokenizer, recv_sys_prompt, completions, number, **kwargs):
  res = []
  for comp, num in zip(completions, number):
    recv_prompt = f'state the number from the following description: {comp}'
    recv_input = mk_input(model, tokenizer, recv_sys_prompt, recv_prompt)
    recv_msg = inference(model, tokenizer, recv_input)
    decoded_num = parse_recv_msg(recv_msg)

    reward = 0
    if decoded_num == num:
      reward += 10
    reward -= 0.1 * len(tokenizer(comp).input_ids)
    res.append(reward)
  return res

def gen_numbers(n_samples):
  from itertools import count, islice
  def has_repeats(n, r):
    return any(str(v)*r in str(n) for v in range(10))
  min_repeats = 4
  lower, upper = 100_000, 100_000_000
  gen = (random.randint(lower, upper) for _ in count())
  filtered = (n for n in gen if has_repeats(n, min_repeats))
  return list(islice(filtered, n_samples))

def mk_dataset(n_samples, tokenizer, sender_sys_prompt):
  # sender_prompt = f'the number is {number}'
  # sender_sys_prompt = 'you are a sender agent. your goal is to describe a number.'
  # recv_prompt = f'the sender sent the message: {sender_msg}'
  # recv_sys_prompt = 'you are a receiver agent. your partner, the sender, has sent a message describing a number. your goal is to state the number.'

  prompt = lambda n: f'describe the number {n}'
  numbers = gen_numbers(n_samples)
  # grpo trainer handles this correctly via maybe_apply_chat_template
  prompts = [tokenizer.apply_chat_template(
      mk_prompt(sender_sys_prompt, prompt(n)),
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
    ) for n in numbers]

  return Dataset.from_dict({'prompt': prompts, 'number': numbers})

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
  return checkpoint_commit == get_current_commit()

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
  grpo_config = GRPOConfig(
    # KL to reference model
    beta=0,
    output_dir=checkpoint_path,
    num_generations=4,
    report_to="wandb",
    # log every
    logging_steps=5,
    log_completions=True,
    max_steps=1_000,
    save_steps=100,
    # default 8
    per_device_train_batch_size=32,
    # question-level difficulty bias
    scale_rewards=False,
    top_p=1.0,
    temperature=1.5,
  )
  sys_prompt = 'you are an AI agent performing tasks the user asks you to do'
  def reward_func(completions, number, **kwargs):
    return rewards(model=model, tokenizer=tokenizer, recv_sys_prompt=sys_prompt, completions=completions, number=number, **kwargs)

  grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=mk_dataset(5_000, tokenizer, sys_prompt),
    reward_funcs=reward_func,
    args=grpo_config,
    callbacks=[CustomCheckpointCallback()]
  )
  grpo_trainer.train(resume_from_checkpoint=should_resume(checkpoint_path, override=False))


main()
