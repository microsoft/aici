FROM nvcr.io/nvidia/pytorch:22.12-py3

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN apt-get update && apt-get -y install strace wabt \
linux-tools-common linux-tools-generic linux-tools-`uname -r`

RUN curl -L https://github.com/WebAssembly/binaryen/releases/download/version_116/binaryen-version_116-x86_64-linux.tar.gz \
    | tar zxf - --strip-components=1  -C /usr/local

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add wasm32-unknown-unknown
