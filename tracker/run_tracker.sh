#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_EXE="${SCRIPT_DIR}/build/tracker"

if [ ! -f "$TRACKER_EXE" ]; then
    echo "Error: Tracker executable not found at: $TRACKER_EXE"
    echo "Please build the project first:"
    echo "  cd $SCRIPT_DIR"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j"
    exit 1
fi

if [ ! -x "$TRACKER_EXE" ]; then
    echo "Error: Tracker executable is not executable: $TRACKER_EXE"
    echo "Trying to fix permissions..."
    chmod +x "$TRACKER_EXE" || exit 1
fi

export LD_LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [video_path] [options]"
    echo ""
    echo "Arguments:"
    echo "  video_path          Video file to track (default: data/car.mp4)"
    echo ""
    echo "Options:"
    echo "  --cpu               Use CPU mode (OpenCV matchTemplate)"
    echo "  --shared            Use shared memory CUDA kernel"
    echo "  --const             Use constant memory CUDA kernel"
    echo "  --const_tiled       Use tiled constant memory CUDA kernel"
    echo "  --record            Record tracking to video file (tracked_output.mp4)"
    echo "  --batch=N           Use batched CUDA mode with batch size N"
    echo ""
    exit 0
fi

exec "$TRACKER_EXE" "$@"

