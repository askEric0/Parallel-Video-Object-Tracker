#!/bin/bash
# Wrapper script to run tracker with correct library paths
# Prioritizes system libstdc++ over fslpython's version

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set LD_LIBRARY_PATH to prioritize system GCC libraries
export LD_LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Run the tracker executable
exec "${SCRIPT_DIR}/build/tracker" "$@"

