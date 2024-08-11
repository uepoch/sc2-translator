import os
import subprocess
import typer


from .mpq import (
    add_file_to_mpq,
    fetch_file_from_mpq,
    list_file_from_mpq,
    remove_file_from_mpq,
)
import tempfile

from os import path
from rich import print


def main(
    sc2mod_file: str,
    yes: bool = True,
    mpq_extractor_path: str = os.getenv("MPQ_EXTRACTOR_PATH"),
):
    bad_key = "enUS.SC2Data/LocalizedData/GameStrings.txt"
    good_key = "enUS.SC2Data\\LocalizedData\\GameStrings.txt"

    if any(
        key == "enUS.SC2Data/LocalizedData/GameStrings.txt"
        for key in list_file_from_mpq(mpq_extractor_path, sc2mod_file)
    ):
        print(f"[red]Found bad key in {sc2mod_file}[/red]")
        if not yes and not typer.confirm("Do you want to fix it?"):
            print("[red]Exiting.[/red]")
            exit(1)
        with tempfile.TemporaryDirectory() as temp_dir:
            fetch_file_from_mpq(mpq_extractor_path, sc2mod_file, bad_key, temp_dir)
            remove_file_from_mpq(mpq_extractor_path, sc2mod_file, bad_key)
            add_file_to_mpq(
                mpq_extractor_path, sc2mod_file, path.join(temp_dir, bad_key), good_key
            )
        print(f"[green]Fixed {sc2mod_file}[/green]")
    else:
        print(f"[green]No bad key found in {sc2mod_file}[/green]")


def _cli_main():
    typer.run(main)
