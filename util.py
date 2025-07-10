import wandb
from timeit import default_timer as timer
from functools import wraps


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
