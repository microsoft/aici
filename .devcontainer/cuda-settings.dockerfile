FROM nvcr.io/nvidia/pytorch:23.09-py3

# A100:
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV CUDA_COMPUTE_CAP="80"

# candle is slow without caching; pytorch image has this on by default
ENV CUDA_CACHE_DISABLE=""

ENV LIBTORCH_USE_PYTORCH="1"
ENV LIBTORCH_BYPASS_VERSION_CHECK="1"

# the .so file seems to be missing
RUN ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib/libcudart.so{.*,}
