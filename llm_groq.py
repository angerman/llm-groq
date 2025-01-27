import click
import httpx
import json
import llm
from groq import Groq, AsyncGroq
from pydantic import Field
from typing import Optional, List, Union

MODEL_ENDPOINT = "https://api.groq.com/openai/v1/models"

@llm.hookimpl
def register_models(register):
    models = get_model_details()
    for model in models:
        model_id = model["id"]
        register(LLMGroq(model_id), LLMAsyncGroq(model_id))


def refresh_models():
    user_dir = llm.user_dir()
    groq_models = user_dir / "groq_models.json"
    key = llm.get_key("", "groq", "LLM_GROQ_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'groq' key or the LLM_GROQ_KEY environment variable."
        )
    response = httpx.get(
        MODEL_ENDPOINT,
        headers={"Authorization": f"Bearer {key}"},
    )
    response.raise_for_status()
    models = response.json()
    groq_models.write_text(json.dumps({"data": models["models"]}, indent=2))
    return models["models"]


def get_model_details():
    user_dir = llm.user_dir()
    default_models = {"data": []}
    groq_models = user_dir / "groq_models.json"
    if groq_models.exists():
        models = json.loads(groq_models.read_text())
        return models.get("data", [])
    elif llm.get_key("", "groq", "LLM_GROQ_KEY"):
        try:
            return refresh_models()
        except httpx.HTTPStatusError:
            return default_models["data"]
    else:
        return default_models["data"]


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def groq():
        "Commands relating to the llm-groq plugin"

    @groq.command()
    def refresh():
        "Refresh the list of available Groq models"
        user_dir = llm.user_dir()
        groq_models = user_dir / "groq_models.json"
        try:
            models = refresh_models()
        except httpx.HTTPStatusError:
            click.echo("Failed to refresh models", err=True)
            return

        previous = set(json.loads(groq_models.read_text())["data"]) if groq_models.exists() else set()
        current = {model["id"] for model in models}
        added = current - previous
        removed = previous - current

        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)

        if added or removed:
            click.echo("New list of models:", err=True)
            for model in models:
                click.echo(model["id"], err=True)
        else:
            click.echo("No changes", err=True)


class _Options(llm.Options):
    temperature: Optional[float] = Field(
        description=(
            "Controls randomness of responses. Lower values result in more"
            "predictable outputs, while higher values lead to more varied and"
            "creative outputs."
        ),
        ge=0,
        le=1,
        default=None,
    )
    top_p: Optional[float] = Field(
        description=(
            "Controls randomness by considering the top P probability mass."
            "A lower value makes the output more focused but less creative."
        ),
        ge=0,
        le=1,
        default=None,
    )
    max_tokens: Optional[int] = Field(
        description=(
            "The maximum number of tokens to generate in the completion. The"
            "combined prompt and completion tokens must be under 2048."
        ),
        ge=0,
        lt=2049,
        default=None,
    )
    stop: Optional[Union[str, List[str]]] = Field(
        description=(
            "Sequence that stops the generation. Multiple stop sequences can be"
            "specified as a list."
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
            if prev_response.prompt.system and prev_response.prompt.system != current_system:
                messages.append({"role": "system", "content": prev_response.prompt.system})
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and current_system != prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def build_body(self, messages, options):
        return {
            "model": self.model_id,
            "messages": messages,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_tokens,
            "stop": options.stop,
        }

    def set_usage(self, response, usage):
        response.set_usage(
            input=usage.get("prompt_tokens"),
            output=usage.get("completion_tokens"),
        )


class LLMGroq(llm.Model, _Shared):
    can_stream = True
    needs_key = "groq"
    key_env_var = "GROQ_API_KEY"

    Options = _Options

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        messages = self.build_messages(prompt, conversation)
        body = self.build_body(messages, prompt.options)

        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=self.model_id, messages=messages, stream=stream, **prompt.options.dict()
        )

        try:
            if stream:
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield resp.choices[0].message.content
        except AttributeError as e:
            if "NoneType" in str(e):
                response = resp.choices[0].message
                yield response.content
            else:
                raise e
        finally:
            if getattr(resp, "usage", None):
                self.set_usage(response, resp.usage)


class LLMAsyncGroq(llm.AsyncModel, _Shared):
    can_stream = True
    needs_key = "groq"
    key_env_var = "GROQ_API_KEY"

    Options = _Options

    async def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        messages = self.build_messages(prompt, conversation)
        options = prompt.options.dict()

        client = AsyncGroq(api_key=key)
        resp = await client.chat.completions.create(
            model=self.model_id, messages=messages, stream=stream, **options
        )

        try:
            if stream:
                async for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield resp.choices[0].message.content
        except AttributeError as e:
            if "NoneType" in str(e):
                response = resp.choices[0].message
                yield response.content
            else:
                raise e
        finally:
            if getattr(resp, "usage", None):
                self.set_usage(response, resp.usage)
