#!/bin/bash

# Make sure you have cmake installed
git submodule init && git submodule update
cd MPQExtractor
git submodule init && git submodule update
mkdir -p build && cd build
cmake ..
make
echo "Run the following command to set the MPQ_EXTRACTOR_PATH environment variable:"
echo "export MPQ_EXTRACTOR_PATH=\"$PWD/bin/MPQExtractor\""