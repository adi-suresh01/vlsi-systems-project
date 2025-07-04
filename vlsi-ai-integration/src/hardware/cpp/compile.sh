#!/bin/bash

echo "Compiling C++ Hardware Simulator..."

# Check if g++ is available
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found! Install with: brew install gcc"
    exit 1
fi

# Compile with optimizations
g++ -std=c++17 -O3 -fPIC -shared \
    -pthread \
    -march=native \
    -mtune=native \
    hardware_simulator.cpp \
    -o libhardware_simulator.so

if [ $? -eq 0 ]; then
    echo "C++ compilation successful!"
    echo "Library: libhardware_simulator.so"
    ls -la libhardware_simulator.so
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ready to use C++ hardware simulation!"