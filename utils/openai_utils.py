import openai, dataclasses, logging, time, copy, os
from typing import Union, Optional, Sequence

TOTAL_TOKENS = PROMPT_TOKENS = COMPLETION_TOKENS = 0

OPENAI_KEYS = os.getenv("OPENAI_API_KEY", None)
if OPENAI_KEYS is not None:
    openai.api_key = OPENAI_KEYS
else:
    raise ValueError("Please set OPENAI_API_KEY in your environment variables.")

def get_total_tokens():
    return TOTAL_TOKENS

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

def openai_completion(
    prompt: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments = None,
    model_name="gpt-3.5-turbo",
    sleep_time=60,
    return_text=False,
    return_list=False,
    **decoding_kwargs,
):
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    global TOTAL_TOKENS, PROMPT_TOKENS, COMPLETION_TOKENS
    import openai

    assert not model_name.startswith("gpt-4"), "GPT-4 is tooooo expensive!!!!! Be careful!"
    chat_model = model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4")
    API = openai.ChatCompletion if chat_model else openai.Completion

    if decoding_args is None:
        decoding_args = OpenAIDecodingArguments()

    batch_decoding_args = copy.deepcopy(decoding_args)
    while True:
        try:
            shared_kwargs = dict(
                model=model_name,
                **batch_decoding_args.__dict__,
                **decoding_kwargs,
            )
            if chat_model:
                shared_kwargs["messages"] = prompt
            else:
                shared_kwargs["prompt"] = prompt
            completion_batch = API.create(**shared_kwargs)
            choices = completion_batch.choices

            usage = completion_batch.usage
            TOTAL_TOKENS += usage["total_tokens"]
            PROMPT_TOKENS += usage["prompt_tokens"]
            COMPLETION_TOKENS += usage["completion_tokens"]

            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        choices = [choice.message.content if chat_model else choice.text for choice in choices]
    if decoding_args.n == 1 and not return_list:
        choices = choices[0]
    return choices