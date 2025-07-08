import os
import random
import torch
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
from peft import LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import wandb

def load_model():
  device = 'auto'
  # maybe we have to use a base model here?
  # (eg. Qwen/Qwen3-0.6B-Base)
  # but it generates chinese text...
  # model_id = 'Qwen/Qwen3-0.6B'
  model_id = 'Qwen/Qwen3-8B'

  # qwen has endoftext as pad token
  tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto',
    device_map=device,
  )

  lora_rank = 256
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
  print(f'{full_prompt=}')
  chat = tokenizer.apply_chat_template(
    full_prompt,
    tokenize=False,
    # turn this off and do continue_final_message=True for continuing an assistant message
    add_generation_prompt=True,
    enable_thinking=False,
  )
  print(f'{chat=}')
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
  print(f"called rewards, got: {completions=}, {number=}")
  res = []
  for comp, num in zip(completions, number):
    recv_prompt = f'state the number from the following description: {comp}'
    recv_input = mk_input(model, tokenizer, recv_sys_prompt, recv_prompt)
    print("before recv")
    recv_msg = inference(model, tokenizer, recv_input)
    print("after recv")
    decoded_num = parse_recv_msg(recv_msg)

    reward = 0
    if decoded_num == num:
      reward += 10
    reward -= 0.1 * len(tokenizer(comp).input_ids)
    res.append(reward)
  print(f"done computing reward {res=}")
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


def main():
  os.environ['WANDB_PROJECT'] = 'llm-comm-opt'
  wandb.init(config={"slurm_job_id": os.environ.get("SLURM_JOB_ID")})
  model, tokenizer = load_model()
  grpo_config = GRPOConfig(
    # KL to reference model
    beta=0,
    output_dir="checkpoints",
    num_generations=4,
    report_to="wandb",
    # log every
    logging_steps=5,
    log_completions=True,
    max_steps=1_000,
    # default 8
    per_device_train_batch_size=32,
    # question-level difficulty bias
    scale_rewards=False,
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
  )
  grpo_trainer.train()


main()
