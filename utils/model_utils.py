class ChatBot:
    """
    Only support completion now.
    """

    bot = None
    model_name = None
    port = None
    tokenizer = None
    headless = False

    chat_cnt = 0
    prompt_length = []
    output_length = []
    token_length = []
    use_api = False

    MODEL_TYPE = {"browsing": "Browsing\nALPHA", "gpt3": "Default (GPT-3.5)", "gpt3-old": "Legacy (GPT-3.5)", "gpt4": "GPT-4"}

    @classmethod
    def init(cls, model_name):
        cls.model_name = model_name
        cls.bot = None
        cls.chat_cnt = 0
        print("ChatBot is ready now!")

    @classmethod
    def call_chat_gpt(cls, prompt, **kwargs):
        from .openai_utils import openai_completion

        decoding_args = kwargs.pop("decoding_args", None)
        return_list = kwargs.pop("return_list", False)

        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        return openai_completion(prompt, decoding_args, return_text=True, return_list=return_list)
