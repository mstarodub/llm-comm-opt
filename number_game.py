from datasets import Dataset
import random
from itertools import islice
from util import mk_prompt, mk_inputs_batch, inference_batch


def parse_recv_msg(msg):
  return ''.join(ch for ch in msg if ch.isdigit())


def gen_numbers(n_samples):
  min_repeats = 15
  n_digits = 22
  def gen():
    while True:
      d = str(random.randrange(10))
      p = random.randrange(n_digits - min_repeats + 1)
      if p == 0 and d == '0':
        d = str(random.randrange(1, 10))
      num = [''] * n_digits
      num[p:p + min_repeats] = [d] * min_repeats
      for i in range(n_digits):
        if num[i] == '':
          if i == 0:
            num[i] = str(random.randrange(1, 10))
          else:
            num[i] = str(random.randrange(10))
      yield ''.join(num)
  return list(islice(gen(), n_samples))


def rewards(model, tokenizer, recv_sys_prompt, recv_prompt_func, recv_msg_parse_func, completions, number, **kwargs):
  recv_prompts = [recv_prompt_func(comp) for comp in completions]
  recv_inputs = mk_inputs_batch(model, tokenizer, recv_sys_prompt, recv_prompts)
  recv_msgs = inference_batch(model, tokenizer, recv_inputs)
  res = []
  for comp, num, recv_msg in zip(completions, number, recv_msgs):
    decoded_num = recv_msg_parse_func(recv_msg)
    reward = 0
    if decoded_num == num:
        reward += 10
    reward -= 0.1 * len(tokenizer(comp).input_ids)
    res.append(reward)
  return res


def gen_dataset(sys_prompt, prompt_func, n_samples, tokenizer, numbers):
  # grpo trainer handles this correctly via maybe_apply_chat_template
  prompts = [tokenizer.apply_chat_template(
    mk_prompt(sys_prompt, prompt_func(n)),
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
  ) for n in numbers]
  return Dataset.from_dict({'prompt': prompts, 'number': numbers})


class NumbersgameExperiment:
  def __init__(self):
    self.sys_prompt = 'you are an AI agent performing tasks the user asks you to do'

  def mk_dataset(self, n_samples, tokenizer):
    numbers = gen_numbers(n_samples)
    return gen_dataset(
      self.sys_prompt,
      lambda n: f'describe the number {n}',
      n_samples,
      tokenizer,
      numbers
    )

  def mk_reward_func(self, model, tokenizer):
    def reward_func(completions, number, **kwargs):
      return rewards(
        model=model,
        tokenizer=tokenizer,
        recv_sys_prompt=self.sys_prompt,
        recv_prompt_func=lambda comp: f'state the number from the following description: {comp}',
        recv_msg_parse_func=parse_recv_msg,
        completions=completions,
        number=number,
        **kwargs
      )
    return reward_func


def gen_binary_numbers(n_samples, n_bits=20):
  return [
    ''.join(random.choice('01') for _ in range(n_bits))
    for _ in range(n_samples)
  ]


def parse_recv_msg_binary(msg):
  return ''.join(ch for ch in msg if ch in '01')


class BinaryNumbersgameExperiment:
  def __init__(self):
    self.sys_prompt = 'you are an AI agent performing tasks the user asks you to do'

  def mk_dataset(self, n_samples, tokenizer):
    numbers = gen_binary_numbers(n_samples)
    return gen_dataset(
      self.sys_prompt,
      lambda n: f'describe the binary number {n}',
      n_samples,
      tokenizer,
      numbers
    )

  def mk_reward_func(self, model, tokenizer):
    def reward_func(completions, number, **kwargs):
      return rewards(
        model=model,
        tokenizer=tokenizer,
        recv_sys_prompt=self.sys_prompt,
        recv_prompt_func=lambda comp: f'state the binary number from the following description: {comp}',
        recv_msg_parse_func=parse_recv_msg_binary,
        completions=completions,
        number=number,
        **kwargs
      )
    return reward_func
