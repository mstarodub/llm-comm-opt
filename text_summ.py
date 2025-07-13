from datasets import load_dataset


class TextsumExperiment:
  def __init__(self):
    self.sys_prompt = 'you are an AI agent performing tasks the user asks you to do'

  def mk_dataset(self, n_samples, tokenizer)
    # creates an IterableDataset
    # ds = load_dataset(
    #   'roneneldan/TinyStories',
    #   streaming=True,
    #   cache_dir=dataset_dir,
    # )
    # dataset = dataset.map(function_taking_and_ret_dict, remove_columns=['col1'])
    # dataset = dataset.remove_columns('timestamp')
    #
    # trainer.accelerator.register_for_checkpointing(dataset)
    pass

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
        **kwargs,
      )

    return reward_func

def custom_loop(trainer):
  trainer.is_in_train = True
