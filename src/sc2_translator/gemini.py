from datetime import timedelta
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn


import os
import asyncio


from sc2_translator import TRANSLATE_BASE_INPUTS, clean_dataset, lines_to_dict
import google.generativeai as genai
from google.generativeai import caching
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from rich import print
from ratelimit import limits, sleep_and_retry

# Experimental runner with Gemini-Flash, requires Google Cloud API key

API_KEY = os.getenv("GEMINI_API_KEY")

MAX_TOKENS = 8192
model_name = "models/gemini-1.5-flash-001"
BASIC_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": MAX_TOKENS,
    "top_p": 0.97,
    "top_k": 64,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

SYSTEM_INSTRUCTION = TRANSLATE_BASE_INPUTS[0]["content"]


genai.configure(api_key=API_KEY)


def model_for_input(
    original_lines: list[(str, str)],
) -> tuple[genai.GenerativeModel, caching.CachedContent | None]:
    print("Creating model for input")
    try:
        cache = caching.CachedContent.create(
            model=model_name,
            system_instruction=SYSTEM_INSTRUCTION,
            contents=[
                "Original Lines:" "\n".join(f"{k}: {v}" for k, v in original_lines)
            ],
        )
        return genai.GenerativeModel.from_cached_content(
            cache, generation_config=BASIC_CONFIG, safety_settings=SAFETY_SETTINGS
        ), cache
    except Exception as e:
        print(
            f"[red]Failed to create cached content, creating model from scratch: {e}[/red]"
        )
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=BASIC_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=SYSTEM_INSTRUCTION
            + f"\n\n{'\n'.join(f'{k}: {v}' for k, v in original_lines)}",
        ), None


def batchify_lines(original_lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    # We have a MAX_TOKENS limit for the model. We need to batchify the lines so that we don't exceed the limit at best when outputting the lines.
    # We will return the first and last key-value pair for each batch.
    # We assume the byte length of the translated value is equivalent to the reference value.
    batches = []
    cur_min = original_lines[0]
    cur_max = original_lines[0]
    cur_len = 0
    new_batch = False
    for k, v in original_lines:
        # If we would exceed the MAX_TOKENS, we need to start a new batch.
        if new_batch:
            cur_min = k
            new_batch = False
        if cur_len + len(k) + len(v) > MAX_TOKENS:
            batches.append((cur_min, cur_max))
            new_batch = True
            cur_len = len(k) + len(v)
        else:
            cur_len += len(k) + len(v)
        cur_max = k
    batches.append((cur_min, cur_max))
    return batches

SEM = asyncio.Semaphore(50)


async def execute_query(
    model: genai.GenerativeModel, batch: tuple[str, str]
) -> list[str]:
    global SEM
    async with SEM:
        resp = await model.generate_content_async(
            contents=f"Translate the lines between {batch[0]} and {batch[1]} with KEY=VALUE pairs."
        )
        return resp.text.splitlines()


async def process_batch(
    model: genai.GenerativeModel,
    batch: tuple[str, str],
    progress: Progress,
    task_id: TaskID,
) -> list[str]:
    resp = await execute_query(model, batch)
    progress.update(task_id, advance=1)
    return resp


async def translate_lines(original_lines: dict[str, str], cached_lines: dict[str, str]) -> dict[str, str]:
    all_lines = sorted(original_lines.items(), key=lambda x: x[0].split("/")[-1])
    results = []
    output = dict()

    remaining_lines = all_lines
    if cached_lines:
        remaining_lines = list(clean_dataset(
                    {l[0]: l[1] for l in all_lines}, cached_lines, True
                ).items())
        output.update(cached_lines)
    while len(remaining_lines) > 0:
        try:
            model, cache = model_for_input(remaining_lines)
            batches = batchify_lines(remaining_lines)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[progress.remaining]{task.remaining} batches remaining"),
                TextColumn("[progress.elapsed]{task.elapsed:.2f}s"),
            ) as progress:
                task = progress.add_task(
                    "[cyan]Translating batches...", total=len(batches)
                )
                tasks = [
                    process_batch(model, batch, progress, task) for batch in batches
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if not isinstance(result, Exception):
                    output.update(lines_to_dict(result))
            for result in results:
                if isinstance(result, Exception):
                    print(f"[red]Error: {result}[/red]")
                    raise result
            remaining_lines = list(clean_dataset(
                    {l[0]: l[1] for l in all_lines}, output, True
                ).items())

        except Exception as e:
            print(f"[red]Error: {e}[/red]")
            raise e
            return output
        finally:
            if cache:
                cache.delete()
            for c in caching.CachedContent.list():
                c.delete()
    return output


def _main():
    import typer

    typer.run(main)


def main(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    try:
        with open(output_file, "r") as f:
            cache = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        cache = {}
    resp = asyncio.run(translate_lines(lines_to_dict(lines), lines_to_dict(cache)))
    with open(output_file, "w+") as f:
        f.write("\n".join(f"{k}={v}" for k, v in resp.items()))
    print(f"[green]Translated {len(resp)} lines to {output_file}[/green]")
