import pathlib
import shutil
import subprocess
import tempfile
import os

from rich import print

from .tools import normalize_path


from os import path


def mpq_copy_file(mpq_extractor_path, sc2mod_file, src: str, dst: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        fetch_file_from_mpq(mpq_extractor_path, sc2mod_file, src, tmpdir)
        add_file_to_mpq(
            mpq_extractor_path,
            sc2mod_file,
            pathlib.Path(tmpdir) / pathlib.Path(normalize_path(src)).name,
            dst,
        )


def is_mod_folder(modpath: str) -> bool:
    return modpath.strip("/").endswith((".SC2Mod", ".SC2Map")) and path.isdir(modpath)


def mpq_has_file(mpq_extractor_path, sc2mod_file, file: str) -> bool:
    print_warning_sc2mod_folder()
    return file in list_file_from_mpq(mpq_extractor_path, sc2mod_file)


def list_file_from_mpq(mpq_extractor_path, sc2mod_file) -> list[str]:
    if is_mod_folder(sc2mod_file):
        print_warning_sc2mod_folder()
        return [
            str(p).removeprefix(sc2mod_file).replace("/", "\\")
            for p in pathlib.Path(sc2mod_file).rglob("*.*")
            if p.is_file()
        ]
    else:
        with tempfile.NamedTemporaryFile() as temp_file:
            mpq_extractor_run(mpq_extractor_path, [sc2mod_file, "-l", temp_file.name])
            with open(temp_file.name, "r") as f:
                return [line.strip() for line in f.readlines()]


def remove_file_from_mpq(mpq_extractor_path, sc2mod_file, file: str):
    if is_mod_folder(sc2mod_file):
        print_warning_sc2mod_folder()
        # Handle removing file from SC2Mod folder
        os.remove(path.join(sc2mod_file, file))
    else:
        mpq_extractor_run(mpq_extractor_path, [sc2mod_file, "--rm", file])


def add_file_to_mpq(extractor_path: str, sc2mod_file: str, file: str, path: str):
    file: pathlib.Path = pathlib.Path(file)
    print(f"Adding {file} to {sc2mod_file}")
    if is_mod_folder(sc2mod_file):
        print_warning_sc2mod_folder()
        folder = pathlib.Path(sc2mod_file)
        out_file = folder / normalize_path(path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Handle adding file to SC2Mod folder
        shutil.copyfile(file, out_file)
    else:
        mpq_extractor_run(extractor_path, [sc2mod_file, "--addfile", f"{file}={path}"])


def fetch_file_from_mpq(extractor_path: str, sc2mod_file: str, file: str, outpath: str):
    print(f"Fetching {file} from {sc2mod_file}")
    if is_mod_folder(sc2mod_file):
        print_warning_sc2mod_folder()
        file = normalize_path(file)
        shutil.copyfile(path.join(sc2mod_file, file), pathlib.Path(outpath) / file.name)
    else:
        mpq_extractor_run(extractor_path, [sc2mod_file, "-e", file, "-o", outpath])


def mpq_extractor_run(mpq_extractor_path: str, cmds: list[str]):
    if not mpq_extractor_path or not os.path.exists(mpq_extractor_path):
        print(f"[red]MPQ extractor not found at {mpq_extractor_path}[/red]")
        exit(1)
    try:
        result = subprocess.run(
            [mpq_extractor_path, *cmds], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[red]Error running MPQ extractor: {e}[/red]")
        print(f"Command output:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise


WARNED = False


def print_warning_sc2mod_folder():
    global WARNED
    if not WARNED:
        print(
            f"[yellow]Warning: You are using a SC2Mod folder, which support is experimental.[/yellow]"
        )
        WARNED = True
