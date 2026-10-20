#!/bin/bash

export THE_ROCK=/data/rocm/therock-7.14/install

unset HSA_OVERRIDE_GFX_VERSION
unset ROCM_PATH
unset HIP_PATH
unset HIP_PLATFORM

# Preserve the active Python virtual environment.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/bin" ]; then
    export PATH="$THE_ROCK/bin:$THE_ROCK/lib/llvm/bin:$VIRTUAL_ENV/bin:/usr/bin:/bin"
else
    export PATH="$THE_ROCK/bin:$THE_ROCK/lib/llvm/bin:/usr/bin:/bin"
fi

# Keep The Rock libraries first, but don't accumulate duplicates.
export LD_LIBRARY_PATH="$THE_ROCK/lib:$THE_ROCK/lib/llvm/lib"
