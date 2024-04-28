import functools
import json
import logging
from typing import Any, Literal, Optional, cast

import backoff
import erniebot

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM
import time
import os

ERRORS = erniebot.errors

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class ERNIE(LM):
    """Wrapper around OpenAI's GPT API.

    Args:
        model (str, optional): OpenAI supported LLM model to use. Defaults to "gpt-3.5-turbo-instruct".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai"], optional): The API provider to use. Defaults to "openai".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """
    

    def __init__(
        self,
        model: str = "ernie-3.5",
        access_token: Optional[str] = "",
        api_base: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        erniebot.api_type = 'aistudio'
        if not access_token:
            access_token = os.environ.get("EB_ACCESS_TOKEN")
        erniebot.access_token = access_token
        super().__init__(model)

        self.provider = "ernie"
        self.system_prompt = system_prompt


        default_model_type = "chat"
        # self.model_type = model_type if model_type else default_model_type
        self.model_type = default_model_type

        self.kwargs = {
            "temperature": 0.7,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"OpenAI Response Token Usage: {total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            kwargs["messages"] = messages
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = chat_request(**kwargs)
        else:
            kwargs["prompt"] = prompt
            response = completions_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retreival of GPT-3 completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)


    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from GPT-3.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        self.log_usage(response)

        completions = [response["result"]]
        return completions


@CacheMemory.cache
def cached_gpt3_request_v2(**kwargs):
    return erniebot.ChatCompletion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_request_v2_wrapped(**kwargs):
    return cached_gpt3_request_v2(**kwargs)



@CacheMemory.cache
def v1_cached_gpt3_request_v2(**kwargs):
    return erniebot.ChatCompletion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def v1_cached_gpt3_request_v2_wrapped(**kwargs):
    return v1_cached_gpt3_request_v2(**kwargs)


@CacheMemory.cache
def v1_cached_gpt3_turbo_request_v2(**kwargs):
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    # breakpoint()
    # time.sleep(0.1)
    if "n" in kwargs:
        kwargs.pop("n")
    if "temperature" in kwargs and kwargs["temperature"] == 0:
        kwargs.pop("temperature")
    return erniebot.ChatCompletion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs):
    result = v1_cached_gpt3_turbo_request_v2(**kwargs)
    return result


def chat_request(**kwargs):
    return  v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs).to_dict()


def completions_request(**kwargs):
    return v1_cached_gpt3_request_v2_wrapped(**kwargs).to_dict()
