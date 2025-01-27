import click
import httpx
import json
import llm
from groq import Groq, AsyncGroq
from pydantic import Field
from typing import Optional, List, Union

# An internal map of recognized Groq model IDs to their "internal" or "actual"
# model names on the server. By default, if the server returns an "id" that is
# in this dict, we will pass the mapped model name to the .create() calls.
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


@llm.hookimpl
def register_models(register):
    """
    Register models based on the local 'groq_models.json' file if it exists,
    otherwise on first run it uses model_map as defaults. The user can
    'llm groq refresh' to fetch dynamically from the Groq API.
    """
    models = get_model_details()
    for model in models:
        model_id = "groq/{}".format(model["id"])
        register(LLMGroq(model_id), LLMAsyncGroq(model_id))


def refresh_models():
    """
    Fetch the list of available models from Groq's API and store them locally.
    """
    user_dir = llm.user_dir()
    groq_models_path = user_dir / "groq_models.json"
    key = llm.get_key("", "groq", "LLM_GROQ_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'groq' key or the LLM_GROQ_KEY environment variable."
        )
    response = httpx.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    response.raise_for_status()
    data = response.json()
    groq_models_path.write_text(json.dumps(data, indent=2))
    return data


def get_model_details():
    """
    Return details of available Groq models, preferring whatever we have in
    groq_models.json. If that file doesn't exist (and the user has a key), try
    calling refresh_models() once. If we fail, fallback to local model_map keys.
    """
    user_dir = llm.user_dir()
    groq_models_path = user_dir / "groq_models.json"
    # Default fallback if we can't fetch from the server:
    models = {"data": [{"id": m} for m in model_map.keys()]}
    if groq_models_path.exists():
        models = json.loads(groq_models_path.read_text())
    else:
        # Attempt to refresh if a key is present
        if llm.get_key("", "groq", "LLM_GROQ_KEY"):
            try:
                models = refresh_models()
            except httpx.HTTPStatusError:
                pass
    return models["data"]


def get_model_ids():
    """Return just the list of Groq model IDs from get_model_details()."""
    return [model["id"] for model in get_model_details()]


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def groq():
        "Commands relating to the llm-groq plugin"

    @groq.command()
    def refresh():
        """
        Refresh the list of available Groq models from https://api.groq.com/
        and update the local groq_models.json file in your llm user directory.
        """
        before = set(get_model_ids())
        refresh_models()
        after = set(get_model_ids())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_ids():
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)


class _Options(llm.Options):
    temperature: Optional[float] = Field(
        description=(
            "Controls randomness of responses. A lower temperature leads to "
            "more predictable outputs, while a higher temperature results in "
            "more varied and sometimes more creative outputs. As the temperature "
            "approaches zero, the model will become deterministic and repetitive."
        ),
        ge=0,
        le=1,
        default=None,
    )
    top_p: Optional[float] = Field(
        description=(
            "Controls randomness of responses via nucleus sampling. 0.5 means "
            "half of all likelihood-weighted options are considered."
        ),
        ge=0,
        le=1,
        default=None,
    )
    max_tokens: Optional[int] = Field(
        description=(
            "The maximum number of tokens that the model can process in a "
            "single response. This limits ensures computational efficiency "
            "and resource management. Requests can use up to 2048 tokens "
            "shared between prompt and completion."
        ),
        ge=0,
        lt=2049,
        default=None,
    )
    stop: Optional[Union[str, List[str]]] = Field(
        description=(
            "A stop sequence is a predefined text string that signals the AI "
            "to stop generating content, ensuring responses remain focused and "
            "concise. If multiple values are needed, pass an array of strings."
        ),
        default=None,
    )


class _Shared:
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


class LLMGroq(llm.Model, _Shared):
    can_stream = True
    Options = _Options

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        messages = self.build_messages(prompt, conversation)

        actual_model_name = self.model_id.replace("groq/", "")
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            messages=messages,
            model=actual_model_name,
            stream=stream,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            max_tokens=prompt.options.max_tokens,
            stop=prompt.options.stop,
        )
        if stream:
            for chunk in resp:
                if chunk.choices[0] and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield resp.choices[0].message.content


class LLMAsyncGroq(llm.AsyncModel, _Shared):
    can_stream = True
    Options = _Options

    async def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        messages = self.build_messages(prompt, conversation)

        actual_model_name = self.model_id.replace("groq/", "")
        client = AsyncGroq(api_key=key)
        resp = await client.chat.completions.create(
            messages=messages,
            model=actual_model_name,
            stream=stream,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            max_tokens=prompt.options.max_tokens,
            stop=prompt.options.stop,
        )
        if stream:
            async for chunk in resp:
                if chunk.choices[0] and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield resp.choices[0].message.content
