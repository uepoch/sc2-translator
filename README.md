# sc2-cn-translator

Made for the SC2 Modding Discord server.
There's a lot of amazing work made by the CN modding community, but it's not always easy to find the right words or persons to translate.
This tool is made to help with that before someone with actual language skills can look at it and do it properly.

This is a work in progress, and is not yet ready for use.
It uses [Groq](https://groq.com) service to translate the strings, and a modified version of [MPQExtractor](https://github.com/SC2AD/MPQExtractor) to add the translated strings to the SC2 mod.

## Requirements

- Python 3.10+
- CMake (Somewhat recent version)

## Installation

```bash
# Make sure you have `cmake` installed.
# Run `build.sh` to build the MPQExtractor submodule.
./build.sh

# Use your favorite python package manager to install the dependencies.
python -m venv .venv && source .venv/bin/activate
pip install -e .
export MPQ_EXTRACTOR_PATH="MPQExtractor/build/bin/MPQExtractor"

```

The consistency pass is done with the Gemma2 model, and tend to hallucinate some sentences that are not in the original file.
I would recommend to use --no-consistency-pass if you want to avoid this at the cost of some extra wonky unit names.

## Usage

```bash
Usage: sc2-translator [OPTIONS] FILE                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    file      TEXT  [default: None] [required]                                                                                                                                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --groq-api                                       TEXT                                                                            [default: FROM_ENV]                                                                               │
│ --leftovers             --no-leftovers                                                                                           [default: leftovers]                                                                                                                              │
│ --output                                         TEXT                                                                            [default: None]                                                                                                                                   │
│ --auto-continue         --no-auto-continue                                                                                       [default: no-auto-continue]                                                                                                                       │
│ --consistency-pass      --no-consistency-pass                                                                                    [default: consistency-pass]                                                                                                                       │
│ --model                                          [llama-3.1-70b-versatile|llama-3.1-8b-instant|gemma2-9b-it|mixtral-8x7b-32768]  [default: llama-3.1-70b-versatile]                                                                                                                │
│ --mpq-extractor-path                             TEXT                                                                            [default: MPQExtractor/build/bin/MPQExtractor]                                                                                                    │
│ --help                                                                                                                           Show this message and exit.                                                                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```