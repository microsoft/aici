FROM nvcr.io/nvidia/pytorch:23.09-py3

# A100:
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV NVTE_FRAMEWORK=pytorch

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# crashes docker?
# RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# takes forever!
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# RUN pip install typing_extensions==4.5.0

RUN apt-get update && apt-get -y install strace wabt \
linux-tools-common linux-tools-generic linux-tools-`uname -r`

RUN curl -L https://github.com/WebAssembly/binaryen/releases/download/version_116/binaryen-version_116-x86_64-linux.tar.gz \
    | tar zxf - --strip-components=1  -C /usr/local

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add wasm32-unknown-unknown
RUN rustup target add wasm32-wasi

RUN cd /tmp && \
    curl -L https://github.com/WebAssembly/wabt/releases/download/1.0.33/wabt-1.0.33.tar.xz | tar Jxf - && \
    cd wabt-1.0.33 && make gcc-release && cp -v bin/wasm-* /usr/bin && cd .. && rm -rf wabt-1.0.33

RUN pip install -U flash-attn

# need git 2.41 for GCM/Github EMU account switching
# https://askubuntu.com/questions/568591/how-do-i-install-the-latest-version-of-git-with-apt
RUN apt-get install -y software-properties-common
RUN apt-add-repository ppa:git-core/ppa
RUN apt-get update && apt-get install -y git

RUN apt-get install -y llvm-dev libclang-dev clang ccache
ENV CUDA_COMPUTE_CAP="80"
# candle is slow without caching; pytorch image has this on by default
ENV CUDA_CACHE_DISABLE=""
ENV LIBTORCH_USE_PYTORCH="1"
ENV CCACHE_DEBUG="1"

# the .so file seems to be missing
RUN ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib/libcudart.so{.*,}