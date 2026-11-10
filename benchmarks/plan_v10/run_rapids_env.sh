#!/usr/bin/env bash
# PLAN_V10: run a python script with the RAPIDS (librmm / libcuvs) shared libraries on LD_LIBRARY_PATH.
# LD_LIBRARY_PATH must be set BEFORE the python process starts (glibc reads it at startup).
SITE=/home/mazaiying/trident/.venv/lib/python3.10/site-packages
LIBS=$(ls -d "$SITE"/*/lib64 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-13
export PATH=/usr/local/cuda-13/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.9"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
exec /home/mazaiying/trident/.venv/bin/python "$@"
