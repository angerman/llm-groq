import click
import httpx
import json
import llm
from groq import Groq, AsyncGroq
from pydantic import Field
from typing import Optional, List, Union

MODEL_ENDPOINT = "https://api.groq.com/openai/v1/models"

OLD_ALIASES: dict = {
    "groq-gemma2": "gemma2-9b-it",
    "groq-llama3": "llama3-8b-8192",
    "groq-llama3-70b": "llama3-70b-8192",
    "groq-mixtral": "mixtral-8x7b-32768",
    "groq-llama3.1-8b": "llama-3.1-8b-instant",
    "groq-llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq-kimi-k2": "moonshotai/kimi-k2-instruct",
}
OLD_ALIASES_REVERSE: dict = {v: k for k, v in OLD_ALIASES.items()}


@llm.hookimpl
def register_models(register):
    models = get_model_details()
    for model in models:
        groq_model_id = model["id"]
        model_id = "groq/{}".format(groq_model_id)
        vision = vision = "-vision" in model_id
        aliases = ()
        if groq_model_id in OLD_ALIASES_REVERSE:
            aliases = (OLD_ALIASES_REVERSE[groq_model_id],)
        register(
            LLMGroq(model_id, groq_model_id, vision=vision),
            LLMAsyncGroq(model_id, groq_model_id, vision=vision),
            aliases=aliases,
        )


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
    groq_models.write_text(json.dumps({"models": models["data"]}, indent=2))
    return models["data"]


def get_model_details():
    user_dir = llm.user_dir()
    default_models = {"models": []}
    groq_models = user_dir / "groq_models.json"
    if groq_models.exists():
        models = json.loads(groq_models.read_text())
        return models.get("models", [])
    elif llm.get_key("", "groq", "LLM_GROQ_KEY"):
        try:
            return refresh_models()
        except httpx.HTTPStatusError:
            return default_models["models"]
    else:
        return default_models["models"]


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
        previous = (
            {model["id"] for model in json.loads(groq_models.read_text())["models"]}
            if groq_models.exists()
            else set()
        )
        try:
            models = refresh_models()
        except httpx.HTTPStatusError:
            click.echo("Failed to refresh models", err=True)
            return
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


def _attachment(attachment):
    base64_content = attachment.base64_content()
    url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    return {"type": "image_url", "image_url": {"url": url}}


class _Shared:
    def __init__(self, model_id, groq_model_id, vision=False):
        self.model_id = model_id
        self.groq_model_id = groq_model_id
        self.vision = vision
        if vision:
            self.attachment_types = {
                "image/png",
                "image/jpeg",
                "image/gif",
            }

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            if prompt.attachments:
                messages.append(
                    {
                        "role": "user",
                        "content": [_attachment(a) for a in prompt.attachments]
                        + [{"type": "text", "text": prompt.prompt}],
                    }
                )
            else:
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

            if prev_response.attachments:
                messages.append(
                    {
                        "role": "user",
                        "content": [_attachment(a) for a in prev_response.attachments]
                        + [{"type": "text", "text": prev_response.prompt.prompt}],
                    }
                )
            else:
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
            messages.append({"role": "assistant", "content": prev_response.text_or_raise()})
        if prompt.system and current_system != prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        if prompt.attachments:
            messages.append(
                {
                    "role": "user",
                    "content": [_attachment(a) for a in prompt.attachments]
                    + [{"type": "text", "text": prompt.prompt}],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def set_usage(self, response, usage):
        response.set_usage(
            input=usage.get("prompt_tokens"),
            output=usage.get("completion_tokens"),
            details=usage,
        )


class LLMGroq(llm.Model, _Shared):
    can_stream = True
    needs_key = "groq"
    key_env_var = "GROQ_API_KEY"

    Options = _Options

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        messages = self.build_messages(prompt, conversation)

        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=self.groq_model_id,
            messages=messages,
            stream=stream,
            **prompt.options.dict(),
        )
        usage = None

        try:
            if stream:
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    if chunk.x_groq and chunk.x_groq.usage:
                        usage = chunk.x_groq.usage.model_dump()
            else:
                yield resp.choices[0].message.content
                usage = resp.usage.model_dump()
        except AttributeError as e:
            if "NoneType" in str(e):
                response = resp.choices[0].message
                yield response.content
            else:
                raise e
        finally:
            if usage:
                self.set_usage(response, usage)


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
            model=self.groq_model_id, messages=messages, stream=stream, **options
        )
        usage = None

        try:
            if stream:
                async for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    if chunk.x_groq and chunk.x_groq.usage:
                        usage = chunk.x_groq.usage.model_dump()
            else:
                yield resp.choices[0].message.content
                usage = resp.usage.model_dump()
        except AttributeError as e:
            if "NoneType" in str(e):
                response = resp.choices[0].message
                yield response.content
            else:
                raise e
        finally:
            if usage:
                self.set_usage(response, usage)
