FROM nvcr.io/nvidia/pytorch:22.12-py3

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# takes forever!
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
RUN pip install typing_extensions==4.4.0

RUN apt-get update && apt-get -y install strace wabt \
linux-tools-common linux-tools-generic linux-tools-`uname -r`

RUN curl -L https://github.com/WebAssembly/binaryen/releases/download/version_116/binaryen-version_116-x86_64-linux.tar.gz \
    | tar zxf - --strip-components=1  -C /usr/local

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add wasm32-unknown-unknown

RUN cd /tmp && \
    curl -L https://github.com/WebAssembly/wabt/releases/download/1.0.33/wabt-1.0.33.tar.xz | tar Jxf - && \
    cd wabt-1.0.33 && make gcc-release && cp -v bin/wasm-* /usr/bin && cd .. && rm -rf wabt-1.0.33
