import os
import tempfile
import typer
from typing import List
# This script can fix common translation issues in MPQ files

from rich.prompt import Confirm
from rich import print
from .mpq import (
    fetch_file_from_mpq,
    list_file_from_mpq,
    add_file_to_mpq,
    mpq_has_file,
)

from .tools import sanitize_string, Locale, get_localized_data_path
from .mpq import mpq_has_file, add_file_to_mpq, fetch_file_from_mpq

FILE_NAME = "GameStrings.txt"



def main(
    sc2mod_files : List[str],
    yes: bool = True,
    locale: Locale = Locale.enUS,
    game_object: str = "GameStrings.txt",
    mpq_extractor_path: str = os.getenv("MPQ_EXTRACTOR_PATH"),
):
    if not mpq_extractor_path:
        print(f"[red]MPQ extractor not provided. Use MPQ_EXTRACTOR_PATH env var[/red]")
        exit(1)

    for sc2mod_file in sc2mod_files:
        data_path = get_localized_data_path(locale, game_object)
        file_name = data_path.to_file_path().name

        with tempfile.TemporaryDirectory() as temp_dir: 
            fetch_file_from_mpq(mpq_extractor_path, sc2mod_file, data_path, temp_dir)
            if not os.path.exists(os.path.join(temp_dir, file_name)):
                print(f"[red]File {file_name} not found in {temp_dir}[/red]")
                continue
            print(f"[green]Fetched {game_object} from {sc2mod_file} in {temp_dir}[/green]")

            lines = []
            with open(os.path.join(temp_dir, file_name), "r") as f:
                lines = f.readlines()
            
            new_lines = []  # New list to store updated lines
            l_count = 0
            for line in lines:
                sanitized = sanitize_string(line)

                if sanitized != line:
                    confirm = yes
                    if not yes:  # Check if 'yes' is not passed
                        print(f"[yellow]Original: {line}[/yellow]")
                        print(f"[green]Sanitized: {sanitized}[/green]")
                        confirm = Confirm.ask("[yellow]Do you want to replace it?[/yellow]")
                    if yes or confirm:
                        l_count += 1
                        new_lines.append(sanitized)  # Replace with sanitized line
                    else:
                        new_lines.append(line)  # Keep original if not confirmed
                else:
                    new_lines.append(line)
                
            with open(os.path.join(temp_dir, file_name), "w") as f:
                f.writelines(new_lines)
            
            
            if l_count > 0:
                add_file_to_mpq(mpq_extractor_path, sc2mod_file, os.path.join(temp_dir, file_name), data_path)
                print(f"[green]Updated {game_object} in {sc2mod_file}[/green]")
                print(f"[green]Updated {l_count} lines over {len(lines)}[/green]")
            else:
                print(f"[green]No changes[/green]")



def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()