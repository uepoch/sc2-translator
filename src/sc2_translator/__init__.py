import pathlib
import asyncio
import shutil
import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Confirm
from rich.traceback import install
from groq import AsyncGroq
from typing import Generator
import os
import regex
import subprocess
import tempfile
from enum import Enum
from os import path

from groq import RateLimitError
from sc2_translator.tools import ignore_all_but_txt


class Locale(str, Enum):
    zhCN = "zhCN"
    enUS = "enUS"


def get_localized_data_path(locale: Locale, object_name: str):
    return f"{locale.value}.SC2Data\\LocalizedData\\{object_name}.txt"


class ModelChoice(str, Enum):
    STRONG = "llama-3.1-70b-versatile"
    FAST = "llama-3.1-8b-instant"
    GEMMA = "gemma2-9b-it"
    MIXTRAL = "mixtral-8x7b-32768"


CONCURRENCY_SEM = asyncio.Semaphore(10)


MPQ_EXTRACTOR_PATH = ()
GROQ_API = os.getenv("GROQ_API")

if os.getenv("DEBUG"):
    install(show_locals=True)

CONSISTENCY_BASE_INPUTS = [
    {
        "role": "system",
        "content": """
    You are in LINE mode, and can only output "<KEY>=<VALUE>" lines.
    You will be provided with <KEY>=<VALUE> lines, where VALUE is a localised string for a Starcraft 2 Unit, tooltip or ability.
    Make sure entries associated with the same unit are translated in a consistent way. (e.g. if you see "Avatar" 3 times and "Zelot" once, favor using "Avatar" for all four)

    Do not modify values if they are already in English and consistent with their neighbours.
    You still have to OUTPUT "<KEY>=<VALUE>" lines for each input line you receive.
    DO NOT repeat KEY name in VALUE if it looks like a unit.
    Only output the "<KEY>=<VALUE>" lines.
""",
    },
]

TRANSLATE_BASE_INPUTS = [
    {
        "role": "system",
        "content": """
You will be provided with <KEY>=<VALUE> lines, where VALUE is a Chinese localised string for a Starcraft 2 Unit, spell or ability.

Translate ONLY the chinese graphemes in the VALUE into English, using the context of a Starcraft 2 universe.

DO NOT repeat KEY name in translated VALUE if it looks like a unit.
For abilities names, take inspiration of the english key to help with translation
Ensure the translated value is grammatically correct and makes sense in the context of the Starcraft 2 universe.
Similar Keys often represent the same unit, try to be consistent when translating values for similar keys.
Be consice, avoid unnecessary long sentences.
Use "<n/>" where a new line is appropriate.

Only output the "<KEY>=<VALUE>" lines.
YOU MUST OUTPUT AN EQUAL SIGN IN EVERY SINGLE LINE.
""",
    },
]

MAX_TOKENS = 8000

from .mpq import (
    add_file_to_mpq,
    fetch_file_from_mpq,
    is_mod_folder,
    mpq_copy_file,
    mpq_has_file,
)


async def execute_query(inputs: list[dict[str, str]], model: ModelChoice) -> str:
    assert inputs
    try:
        async with CONCURRENCY_SEM:
            resp = await AsyncGroq(api_key=GROQ_API).chat.completions.create(
                model=model.value,
                messages=inputs,
                max_tokens=MAX_TOKENS,
                stream=False,
                temperature=0.7,
            )
            content = resp.choices[0].message.content
            if os.getenv("DEBUG"):
                print(content)
            return content
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise e


def load_file(filename) -> list[str]:
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(f"Loaded {len(lines)} lines from {filename}")
    return lines


def batchify(lines: list[str]) -> Generator[list[str], None, None]:
    acc_size = 0
    cur = []
    for line in lines:
        acc_size += len(line)
        if acc_size > MAX_TOKENS * 0.8:
            if not cur:
                raise ValueError("Batch size exceeded with no lines")
            yield cur
            cur = []
            acc_size = 0
        cur.append(line)
    if cur:
        yield cur


async def query_batch(
    base_query, batch, progress: Progress, task: TaskID, model: ModelChoice, retry=True
):
    inputs = base_query + [{"role": "user", "content": "\n".join(batch)}]
    try:
        result = await execute_query(inputs, model)
        progress.update(task, advance=len(batch))
        return result
    except RateLimitError as e:
        retry_after = int(e.response.headers.get("retry-after", "30")) + 5
        print(f"[red]Rate limit exceeded, waiting for {retry_after} seconds.[/red]")
        await asyncio.sleep(retry_after)
        return await query_batch(base_query, batch, progress, task, model, retry=False)
    except Exception as e:
        if retry:
            progress.console.print(f"[yellow]Retrying batch due to error: {e}[/yellow]")
            return await query_batch(
                base_query, batch, progress, task, model, retry=False
            )
        else:
            progress.console.print(f"[red]Batch query failed after retry: {e}[/red]")
            progress.update(task, advance=len(batch))
            return None


async def process_batch(
    base_query, lines: dict[str, str], model: ModelChoice
) -> tuple[dict[str, str], list[str]]:
    all_outputs = []
    failed_lines = []
    lines = [f"""{key}={value}""" for key, value in lines.items()]
    batches = list(batchify(lines))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[progress.remaining]{task.remaining} lines remaining"),
    ) as progress:
        task = progress.add_task("[cyan]Processing lines...", total=len(lines))

        tasks = [
            query_batch(base_query, batch, progress, task, model) for batch in batches
        ]
        results = await asyncio.gather(*tasks)

        for batch, result in zip(batches, results):
            if result is not None:
                all_outputs.extend(result.split("\n"))
            else:
                failed_lines.extend(batch)

    all_outputs = lines_to_dict(all_outputs)
    print(
        f"Processed {len(all_outputs)} lines successfully and failed to translate {len(failed_lines)} lines in {len(batches)} batches."
    )
    return all_outputs, failed_lines


def write_file(filename, lines):
    with open(filename, "w") as f:
        f.writelines(line + "\n" for line in lines)


def lines_to_dict(lines: list[str]):
    ret = {}
    for line in lines:
        if line.startswith("//") or line.startswith("#"):
            continue
        try:
            # Remove all non-ascii characters from keys
            key = line.split("=")[0].encode("ascii", "ignore").decode("ascii")
            value = line.split("=", 1)[1].strip()
            ret[key] = value
        except IndexError:
            continue
    return ret


def clean_dataset(lines: dict[str, str], outputs: dict[str, str], do_leftovers: bool):
    missing_keys = set(lines.keys()) - set(outputs.keys())
    remaining = {key: lines[key] for key in missing_keys}
    outputs = {k: v for k, v in outputs.items() if k in lines}
    if len(remaining) > 0:
        print(f"[yellow]Missing {len(remaining)} lines.[/yellow]")
    if do_leftovers:
        # Check if any value still contain chinese characters
        leftovers = {
            key: lines[key]
            for key, value in outputs.items()
            if regex.search(r"\p{Han}", value)
        }
        leftovers.update(
            {key: lines[key] for key, value in outputs.items() if "Original Lines:" in value and key in lines}
        )
        if leftovers:
            print(
                f"[yellow] {len(leftovers)} lines still had invalid characters.[/yellow]"
            )
        remaining.update(leftovers)
    return remaining


async def process_lines(
    lines: dict[str, str],
    base_query,
    process_leftovers,
    auto_continue,
    model: ModelChoice,
):
    # First, sort the lines by last component of the key.
    # Example: Abil/Foo/02, Abil/Bar/03, Abil/Baz/01
    # Sorted by the last component, we get:
    # Abil/Baz/01, Abil/Bar/02, Abil/Foo/03
    sorted_lines = dict(sorted(lines.items(), key=lambda x: x[0].split("/")[-1]))
    outputs, failed_lines = await process_batch(base_query, sorted_lines, model)
    missing = clean_dataset(lines, outputs, process_leftovers)
    while missing:
        print(f"[red] operation is failing for {len(missing)} lines.[/red]")
        if os.getenv("DEBUG"):
            for key, value in missing.items():
                print(f"[red]{key}={value}[/red]")
        if auto_continue or Confirm.ask(
            "Do you want to retry them?", default=True, show_default=False
        ):
            auto_continue = True
            missing_outputs, missing_failed_lines = await process_batch(
                base_query, missing, model
            )
            outputs.update(missing_outputs)
            failed_lines.extend(missing_failed_lines)
            missing = clean_dataset(lines, outputs, process_leftovers)
        else:
            failed_lines.extend([f"{key}={value}" for key, value in missing])
            break
    return outputs, failed_lines


async def translate_file(
    file: str,
    leftovers: bool,
    output: str | None,
    auto_continue: bool,
    consistency_pass: bool,
    model: ModelChoice,
    reference: dict[str, str] | None,
) -> bool:
    if not output:
        output = pathlib.Path(file).stem + "_translated.txt"
    print("[green]Loading original file...[/green]")
    lines = lines_to_dict(load_file(file))

    if reference:
        print(f"[yellow]Ignoring {len(reference)} keys.[/yellow]")
        lines = {key: value for key, value in lines.items() if key not in reference}

    if not lines:
        print("[red]No lines to translate.[/red]")
        return False

    print("[green]Translating...[/green]")
    outputs, failed_lines = await process_lines(
        lines, TRANSLATE_BASE_INPUTS, leftovers, auto_continue, model
    )
    # filter out random trash that is not in the original file
    outputs = {key: value for key, value in outputs.items() if key in lines}
    if consistency_pass:
        print("[green]Consistency pass...[/green]")
        outputs, consistent_failed_lines = await process_lines(
            outputs, CONSISTENCY_BASE_INPUTS, False, auto_continue, ModelChoice.GEMMA
        )
        failed_lines.extend(consistent_failed_lines)

    outputs = {key: value for key, value in outputs.items() if key in lines}
    if reference:
        outputs.update(reference)

    if failed_lines:
        failed_lines = sorted(set(failed_lines))
        write_file("failed.txt", failed_lines)
        print(
            f"[red]Failed to translate {len(failed_lines)} lines. See failed.txt for details.[/red]"
        )
    else:
        print("[green]All lines translated successfully![/green]")
    if output:
        write_file(
            output, [f"""{key}={value}""" for key, value in sorted(outputs.items())]
        )
        print(f"[green]Translated file saved to {output}[/green]")
        return True
    else:
        return False


def main(
    input_sc2mod_files: list[str],
    groq_api: str = GROQ_API,
    leftovers: bool = True,
    output: str = None,
    auto_continue: bool = False,
    consistency_pass: bool = False,
    model: ModelChoice = ModelChoice.STRONG,
    mpq_extractor_path: str = os.getenv("MPQ_EXTRACTOR_PATH"),
    replace: bool = False,
    re_use_existing: bool = True,
):
    if not groq_api:
        print(
            "[red]GROQ_API environment variable not set. Please set it to your GROQ API key. Visit groq.com to get one[/red]"
        )
        exit(1)
    if not mpq_extractor_path:
        print(
            "[red]MPQ_EXTRACTOR_PATH environment variable not set. Please set it to your MPQ Extractor path.[/red]"
        )
        exit(1)
    for input_sc2mod_file in input_sc2mod_files:
        if not path.exists(input_sc2mod_file):
            print(f"[red] {input_sc2mod_file} does not exist.[/red]")
            exit(1)
        if not input_sc2mod_file.rstrip("/").endswith((".SC2Mod", ".SC2Map")):
            print(f"[red] {input_sc2mod_file} is not a SC2Mod file.[/red]")
            exit(1)

    async def main_async():
        for input_sc2mod_file in input_sc2mod_files:
            print(f"[cyan]Processing {input_sc2mod_file}...")
            await process_file(
                input_sc2mod_file,
                mpq_extractor_path,
                output,
                leftovers,
                auto_continue,
                consistency_pass,
                model,
                replace,
                re_use_existing,
            )

    asyncio.run(main_async())


async def process_file(
    input_sc2mod_file,
    mpq_extractor_path,
    output,
    leftovers,
    auto_continue,
    consistency_pass,
    model,
    replace,
    re_use_existing,
):
    is_folder = is_mod_folder(input_sc2mod_file)

    # New Temporary folder
    with tempfile.TemporaryDirectory() as temp_dir:
        if is_folder:
            sc2mod_file = shutil.copytree(
                input_sc2mod_file,
                path.join(temp_dir, pathlib.Path(input_sc2mod_file).name),
                ignore=ignore_all_but_txt,
            )
        else:
            sc2mod_file = shutil.copyfile(
                input_sc2mod_file,
                path.join(temp_dir, pathlib.Path(input_sc2mod_file).name),
            )

        if re_use_existing and mpq_has_file(
            mpq_extractor_path,
            sc2mod_file,
            get_localized_data_path(Locale.enUS, "GameStrings"),
        ):
            fetch_file_from_mpq(
                mpq_extractor_path,
                sc2mod_file,
                get_localized_data_path(Locale.enUS, "GameStrings"),
                temp_dir,
            )
            reference = lines_to_dict(load_file(path.join(temp_dir, "GameStrings.txt")))
        else:
            reference = None

        translated_file = path.join(temp_dir, "translated.txt")

        fetch_file_from_mpq(
            mpq_extractor_path,
            sc2mod_file,
            get_localized_data_path(Locale.zhCN, "GameStrings"),
            temp_dir,
        )
        # Make sure we copy the Hotkeys.txt file
        if mpq_has_file(
            mpq_extractor_path,
            sc2mod_file,
            get_localized_data_path(Locale.zhCN, "GameHotkeys"),
        ):
            mpq_copy_file(
                mpq_extractor_path,
                sc2mod_file,
                get_localized_data_path(Locale.zhCN, "GameHotkeys"),
                get_localized_data_path(Locale.enUS, "GameHotkeys"),
            )
            print("[green]Copied GameHotkeys.txt.[/green]")
        else:
            print("[yellow]No GameHotkeys.txt file found, skipping.[/yellow]")

        if not await translate_file(
            path.join(
                temp_dir,
                get_localized_data_path(Locale.zhCN, "GameStrings").rsplit("\\", 1)[-1],
            ),
            leftovers,
            translated_file,
            auto_continue,
            consistency_pass,
            model,
            reference,
        ):
            return
        add_file_to_mpq(
            mpq_extractor_path,
            sc2mod_file,
            translated_file,
            get_localized_data_path(Locale.enUS, "GameStrings"),
        )

        mod_path = pathlib.Path(input_sc2mod_file)

        output_mod_file = output or path.join(
            os.curdir, mod_path.stem + "_translated" + mod_path.suffix
        )

        if replace:
            if is_folder:
                shutil.copytree(
                    input_sc2mod_file,
                    mod_path.name + ".bak",
                    dirs_exist_ok=True,
                    ignore=ignore_all_but_txt,
                )
                shutil.copytree(sc2mod_file, input_sc2mod_file, dirs_exist_ok=True)
            else:
                shutil.copyfile(input_sc2mod_file, mod_path.name + ".bak")
                shutil.copyfile(sc2mod_file, input_sc2mod_file)
            output_mod_file = input_sc2mod_file
        else:
            if is_folder:
                shutil.copytree(sc2mod_file, output_mod_file, dirs_exist_ok=True)
            else:
                shutil.copyfile(sc2mod_file, output_mod_file)
        print(f"[green]Finished! The file is located at {output_mod_file}[/green]")


def _cli_main():
    typer.run(main)


if __name__ == "__main__":
    _cli_main()
