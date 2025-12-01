#!/bin/bash

# Wrapper to run cpub with the correct OpenCV library path, while avoiding
# LD_LIBRARY_PATH conflicts that cause noisy libtinfo warnings from bash.

# First invocation: re-exec this script in a clean environment with
# LD_LIBRARY_PATH unset, so that bash itself links against the system libs.
if [ -z "${CPUB_WRAPPER_INNER:-}" ]; then
  exec env -u LD_LIBRARY_PATH CPUB_WRAPPER_INNER=1 "$0" "$@"
fi

# Second invocation (inner): now we can safely add the OpenCV path just for
# the cpub binary.
export LD_LIBRARY_PATH=/usr/local/depot/fsl/fslpython/envs/fslpython/lib:${LD_LIBRARY_PATH}
exec "$(dirname "$0")/cpub" "$@"
