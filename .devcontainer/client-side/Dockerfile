FROM ubuntu:22.04

# makes it easier to diagnose ccache issues
ENV CCACHE_DEBUG="1"

# need git 2.41 for GCM/Github EMU account switching
# https://askubuntu.com/questions/568591/how-do-i-install-the-latest-version-of-git-with-apt
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-add-repository ppa:git-core/ppa
RUN apt-get update && apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        libjpeg-dev \
        libpng-dev \
    strace wabt \
    linux-tools-common linux-tools-generic \
    llvm-dev libclang-dev clang ccache apache2-utils git-lfs \
    screen bsdmainutils

RUN pip install pytest pytest-forked ujson posix_ipc

RUN curl -L https://github.com/WebAssembly/binaryen/releases/download/version_116/binaryen-version_116-x86_64-linux.tar.gz \
    | tar zxf - --strip-components=1  -C /usr/local

RUN cd /tmp && \
    curl -L https://github.com/WebAssembly/wabt/releases/download/1.0.33/wabt-1.0.33.tar.xz | tar Jxf - && \
    cd wabt-1.0.33 && make gcc-release && cp -v bin/wasm-* /usr/bin && cd .. && rm -rf wabt-1.0.33

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add wasm32-unknown-unknown
RUN rustup target add wasm32-wasi
