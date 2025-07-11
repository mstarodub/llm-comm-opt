import subprocess
import wandb
from timeit import default_timer as timer
import random
from itertools import islice
from functools import wraps


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
  return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

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
