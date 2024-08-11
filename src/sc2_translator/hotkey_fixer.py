import os
import tempfile
import typer

from rich import print
from .mpq import (
    fetch_file_from_mpq,
    list_file_from_mpq,
    add_file_to_mpq,
    mpq_has_file,
    remove_file_from_mpq,
)

FILE_NAME = "GameHotkeys.txt"

# This is a simple script to import hotkeys from zhCN to enUS
# It is not perfect, but it works for the most part


def has_en_hotkeys(mpq_extractor_path, sc2mod_file) -> bool:
    files = list_file_from_mpq(mpq_extractor_path, sc2mod_file)
    return any(
        file.endswith("enUS.SC2Data\\LocalizedData\\GameHotkeys.txt") for file in files
    )


def fix_en_hotkeys(mpq_extractor_path, sc2mod_file, yes: bool = False):
    if (
        not has_en_hotkeys(mpq_extractor_path, sc2mod_file)
        or yes
        or not typer.confirm("Do you want to fix it?")
    ):
        print(
            f"[green]Replacing enUS.SC2Data\\LocalizedData\\GameHotkeys.txt in {sc2mod_file}[/green]"
        )
        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            fetch_file_from_mpq(
                mpq_extractor_path,
                sc2mod_file,
                "zhCN.SC2Data\\LocalizedData\\GameHotkeys.txt",
                temp_dir,
            )
            add_file_to_mpq(
                mpq_extractor_path,
                sc2mod_file,
                os.path.join(temp_dir, "GameHotkeys.txt"),
                "enUS.SC2Data\\LocalizedData\\GameHotkeys.txt",
            )
            print(f"[green]Fixed {sc2mod_file}[/green]")
    else:
        print(f"[green]Nothing to do[/green]")


def main(
    sc2mod_file: str,
    yes: bool = True,
    mpq_extractor_path: str = os.getenv("MPQ_EXTRACTOR_PATH"),
):
    if not mpq_has_file(
        mpq_extractor_path, sc2mod_file, "zhCN.SC2Data\\LocalizedData\\GameHotkeys.txt"
    ):
        print(
            f"[red]No zhCN.SC2Data\\LocalizedData\\GameHotkeys.txt found in {sc2mod_file}[/red]"
        )
        exit(0)
    fix_en_hotkeys(mpq_extractor_path, sc2mod_file, yes)


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
