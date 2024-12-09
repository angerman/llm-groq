# [LLM](https://llm.datasette.io/) plugin providing access to [Groqcloud](http://console.groq.com) models.
# Base off of llm-mistral (https://github.com/simonw/llm-mistral)
import llm
from groq import Groq
from pydantic import Field
from typing import Optional, List, Union


@llm.hookimpl
def register_models(register):
    register(LLMGroq("groq-llama2"))
    register(LLMGroq("groq-llama3"))
    register(LLMGroq("groq-llama3-70b"))
    register(LLMGroq("groq-llama3.1-8b"))
    register(LLMGroq("groq-llama3.1-70b"))
    register(LLMGroq("groq-llama3.1-405b"))
    register(LLMGroq("groq-mixtral"))
    register(LLMGroq("groq-gemma"))
    register(LLMGroq("groq-gemma2"))
    register(LLMGroq("groq-llama-3.3-70b"))


class LLMGroq(llm.Model):
    can_stream = True

    model_map: dict = {
        "groq-gemma": "gemma-7b-it",
        "groq-gemma2": "gemma2-9b-it",
        "groq-llama2": "llama2-70b-4096",
        "groq-llama3": "llama3-8b-8192",
        "groq-llama3-70b": "llama3-70b-8192",
        "groq-mixtral": "mixtral-8x7b-32768",
        "groq-llama3.1-8b": "llama-3.1-8b-instant",
        "groq-llama3.1-70b": "llama-3.1-70b-versatile",
        "groq-llama3.1-405b": "llama-3.1-405b-reasoning",
        "groq-llama-3.3-70b": "llama-3.3-70b-versatile",
    }

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Controls randomness of responses. A lower temperature leads to"
                "more predictable outputs while a higher temperature results in"
                "more varies and sometimes more creative outputs."
                "As the temperature approaches zero, the model will become deterministic"
                "and repetitive."
            ),
            ge=0,
            le=1,
            default=None,
        )
        top_p: Optional[float] = Field(
            description=(
                "Controls randomness of responses. A lower temperature leads to"
                "more predictable outputs while a higher temperature results in"
                "more varies and sometimes more creative outputs."
                "0.5 means half of all likelihood-weighted options are considered."
            ),
            ge=0,
            le=1,
            default=None,
        )
        max_tokens: Optional[int] = Field(
            description=(
                "The maximum number of tokens that the model can process in a"
                "single response. This limits ensures computational efficiency"
                "and resource management."
                "Requests can use up to 2048 tokens shared between prompt and completion."
            ),
            ge=0,
            lt=2049,
            default=None,
        )
        stop: Optional[Union[str, List[str]]] = Field(
            description=(
                "A stop sequence is a predefined or user-specified text string that"
                "signals an AI to stop generating content, ensuring its responses"
                "remain focused and concise. Examples include punctuation marks and"
                'markers like "[end]".'
                'For this example, we will use ", 6" so that the llm stops counting at 5.'
                "If multiple stop values are needed, an array of string may be passed,"
                'stop=[", 6", ", six", ", Six"]'
            ),
            default=None,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        messages = self.build_messages(prompt, conversation)
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            messages=messages,
            model=self.model_map[self.model_id],
            stream=stream,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            max_tokens=prompt.options.max_tokens,
            stop=prompt.options.stop,
        )
        if stream:
            for chunk in resp:
                try:
                    if chunk.choices[0].delta.content:
                        yield from chunk.choices[0].delta.content
                except TypeError:
                    breakpoint()
        else:
            yield from resp.choices[0].message.content
