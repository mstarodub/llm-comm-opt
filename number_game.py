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


def rewards(model, tokenizer, recv_sys_prompt, completions, number, **kwargs):
  recv_prompts = [f'state the number from the following description: {comp}' for comp in completions]
  recv_inputs = mk_inputs_batch(model, tokenizer, recv_sys_prompt, recv_prompts)
  recv_msgs = inference_batch(model, tokenizer, recv_inputs)
  res = []
  for comp, num, recv_msg in zip(completions, number, recv_msgs):
    decoded_num = parse_recv_msg(recv_msg)
    reward = 0
    if decoded_num == num:
        reward += 10
    reward -= 0.1 * len(tokenizer(comp).input_ids)
    res.append(reward)
  return res


class NumberGameExperiment:
  def __init__(self):
    self.sys_prompt = 'you are an AI agent performing tasks the user asks you to do'

  def mk_dataset(self, n_samples, tokenizer):
    prompt = lambda n: f'describe the number {n}'
    numbers = gen_numbers(n_samples)
    # grpo trainer handles this correctly via maybe_apply_chat_template
    prompts = [tokenizer.apply_chat_template(
      mk_prompt(self.sys_prompt, prompt(n)),
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
    ) for n in numbers]
    return Dataset.from_dict({'prompt': prompts, 'number': numbers})

  def mk_reward_func(self, model, tokenizer):
    def reward_func(completions, number, **kwargs):
      return rewards(
        model=model,
        tokenizer=tokenizer,
        recv_sys_prompt=self.sys_prompt,
        completions=completions,
        number=number,
        **kwargs
      )
    return reward_func
