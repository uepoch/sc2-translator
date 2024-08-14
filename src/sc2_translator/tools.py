from enum import Enum
from os import path
import pathlib


def is_sc2mod(file: str) -> bool:
    """Check if the given file is a SC2Mod file."""
    return pathlib.Path(file).name.endswith(".SC2Mod")


def normalize_path(path: str) -> str:
    """Normalize the given file path by replacing backslashes with forward slashes."""
    return pathlib.Path(path.replace("\\", "/"))

# Sanitize a string to fix common translation issues like </n/>
def sanitize_string(string: str) -> str:
    """Sanitize a string to fix common translation issues like replacing </n/> with <n/>."""
    return string.replace("</n/>", "<n/>")

# Generate a shutil ignore function that ignores all files but .txt files
def ignore_all_but_txt(src, names):
    """Generate a list of files to ignore, allowing only .txt files, .SC2Data files, and LocalizedData."""
    return [
        name
        for name in names
        if not name.endswith(".txt")
        and not name.endswith(".SC2Data")
        and not name == "LocalizedData"
    ]


class Locale(str, Enum):
    """Enumeration for supported locales."""
    zhCN = "zhCN"
    enUS = "enUS"

class MpqPath(str):
    """Class representing a path in MPQ format."""
    
    def __init__(self, path: str):
        """Initialize with a given path and convert it to MPQ format."""
        self.path = path
        self.path = self.to_mpq_path()
    
    def to_file_path(self) -> pathlib.Path:
        """Convert the MPQ path to a file path."""
        return pathlib.Path(self.path.replace("\\", "/"))
    
    def to_mpq_path(self) -> str:
        """Convert the file path to MPQ format."""
        return self.path.replace("/", "\\")

def get_localized_data_path(locale: Locale, object_name: str) -> MpqPath:
    """Generate the localized data path for a given locale and object name."""
    return MpqPath(f"{locale.value}.SC2Data\\LocalizedData\\{object_name}")