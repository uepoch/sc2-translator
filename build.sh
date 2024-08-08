#!/bin/bash

# Make sure you have cmake installed
git submodule init && git submodule update
cd MPQExtractor
git submodule init && git submodule update
mkdir -p build && cd build
cmake ..
make