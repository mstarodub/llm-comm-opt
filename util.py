import subprocess
import wandb
from timeit import default_timer as timer
from functools import wraps


def mk_prompt(sys_prompt, prompt):
  return [
    {
      'role': 'system',
      'content': sys_prompt,
    },
    {'role': 'user', 'content': prompt},
  ]


def mk_inputs_batch(model, tokenizer, sys_prompt, prompts):
  full_prompts = [mk_prompt(sys_prompt, prompt) for prompt in prompts]
  chats = [
    tokenizer.apply_chat_template(
      full_prompt,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
    )
    for full_prompt in full_prompts
  ]
  tokenized = tokenizer(chats, padding=True, return_tensors='pt').to(model.device)
  return tokenized


def inference_batch(model, tokenizer, inputs):
  max_new_tokens = 2048
  # TODO: do_sample=True ?
  generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
  # select everything after input tokens (this has input + output by default)
  input_lengths = inputs.input_ids.shape[1]
  output_ids = generated_ids[:, input_lengths:]
  contents = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
  return [content.strip() for content in contents]


def timeit(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = timer()
    result = func(*args, **kwargs)
    end = timer()
    print(f'{func.__name__} took {end - start:.4f} seconds')
    return result

  return wrapper


def download_media(run_id):
  api = wandb.Api()
  run = api.run(f'mxst-university-of-oxford/llm-comm-opt/{run_id}')
  for file in run.files():
    if file.name.startswith('media/table/completions'):
      file.download()


def get_current_commit():
  return subprocess.run(
    ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True
  ).stdout.strip()


def lora_print_trainable_parameters(model):
  all_params, trainable_params = 0, 0
  for _, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
    f'{trainable_params=} | {all_params=} | trainable: {100 * trainable_params / all_params:.3f}%'
  )


if __name__ == '__main__':
  download_media('yh7ey1iw')
