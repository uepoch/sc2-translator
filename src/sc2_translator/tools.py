from os import path
import pathlib



def is_sc2mod(file: str) -> bool:
    return pathlib.Path(file).name.endswith(".SC2Mod")


def normalize_path(path: str) -> str:
    return pathlib.Path(path.replace("\\", "/"))

# Generate a shutil ignore function that ignores all files but .txt files
def ignore_all_but_txt(src, names):
    return [name for name in names if not name.endswith(".txt") and not name.endswith(".SC2Data") and not name == "LocalizedData"]
