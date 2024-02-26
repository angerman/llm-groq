# [LLM](https://llm.datasette.io/) plugin providing access to [Groqcloud](http://console.groq.com) models.
# Base off of llm-mistral (https://github.com/simonw/llm-mistral)
import llm
from groq import Groq

@llm.hookimpl
def register_models(register):
    register(LLMGroq("groq-llama2"))
    register(LLMGroq("groq-mixtral"))

class LLMGroq(llm.Model):
    model_map: dict = {
        "groq-llama2": "llama2-70b-4096",
        "groq-mixtral": "mixtral-8x7b-32768",
    }
    # class Options(llm.Options):

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
        chat_completion = client.chat.completions.create(
            messages=messages, model=self.model_map[self.model_id]
        )
        yield from chat_completion.choices[0].message.content