import os
import random

# from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import wandb

from util import timeit


@timeit
def load_model():
  device = 'auto'
  # maybe we have to use a base model here? Qwen/Qwen3-0.6B-Base
  # "If you’re using a base model, ensure you have a chat template"
  # but it generates chinese text...
  # qwen has endoftext as pad token
  model_id = 'Qwen/Qwen3-0.6B'
  # FastLanguageModel has issues with my custom reward function.
  # seems to require a "labels" key in the dataset
  # model_id = 'unsloth/Qwen3-0.6B-unsloth-bnb-4bit'
  # model, tokenizer = FastLanguageModel.from_pretrained(
  #   model_name=model_id,
  #   fast_inference=True,
  #   dtype=torch.bfloat16,
  #   load_in_4bit=False,
  #   # TODO: figure out lora
  #   full_finetuning = True
  # )

  tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto',
    device_map=device,
  )
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


@timeit
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


def mk_dataset(n_samples, tokenizer, sender_sys_prompt):
  # sender_prompt = f'the number is {number}'
  # sender_sys_prompt = 'you are a sender agent. your goal is to describe a number.'
  # recv_prompt = f'the sender sent the message: {sender_msg}'
  # recv_sys_prompt = 'you are a receiver agent. your partner, the sender, has sent a message describing a number. your goal is to state the number.'

  prompt = lambda n: f'describe the number {n}'
  numbers = [random.randint(1, 100) for _ in range(n_samples)]
  # grpo trainer handles this correctly via maybe_apply_chat_template
  prompts = [tokenizer.apply_chat_template(
      mk_prompt(sender_sys_prompt, prompt(n)),
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
    ) for n in numbers]

  return Dataset.from_dict({'prompt': prompts, 'number': numbers})

def game_turn(model, tokenizer):
  number = 42

  sys_prompt = 'you are an AI agent performing tasks the user asks you to do'
  sender_prompt = f'describe the number {number}'
  # sender_prompt = f'the number is {number}'
  # sender_sys_prompt = 'you are a sender agent. your goal is to describe a number.'
  sender_input = mk_input(model, tokenizer, sys_prompt, sender_prompt)
  sender_msg = inference(model, tokenizer, sender_input)
  print('SENDER MSG', sender_msg)

  recv_prompt = f'state the number from the following description: {sender_msg}'
  # recv_prompt = f'the sender sent the message: {sender_msg}'
  # recv_sys_prompt = 'you are a receiver agent. your partner, the sender, has sent a message describing a number. your goal is to state the number.'
  recv_input = mk_input(model, tokenizer, sys_prompt, recv_prompt)
  recv_msg = inference(model, tokenizer, recv_input)
  print('RECV MSG', recv_msg)
  print('PARSED', parse_recv_msg(recv_msg))


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


# model, tokenizer = load_model()
# game_turn(model, tokenizer)
main()
