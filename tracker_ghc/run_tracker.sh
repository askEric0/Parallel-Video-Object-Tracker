#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_EXE="${SCRIPT_DIR}/build/tracker"

if [ ! -f "$TRACKER_EXE" ]; then
    echo "Error: Tracker executable not found at: $TRACKER_EXE"
    echo "Please build the project first:"
fi

export LD_LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [video_path] [options]"
    echo ""
    echo "Arguments:"
    echo "  video_path          Video file to track (default: data/car.mp4)"
    echo ""
    echo "Options:"
    echo "  --cpu               CPU mode using OpenCV matchTemplate)"
    echo "  --shared            Shared memory CUDA kernel"
    echo "  --const             Constant memory CUDA kernel"
    echo "  --const_tiled       Tiled constant memory CUDA kernel"
    echo "  --record            Record tracking"
    echo "  --batch=N           Batched CUDA mode with batch size N"
    echo "  --first             Select template from first frame"
    exit 0
fi

exec "$TRACKER_EXE" "$@"

