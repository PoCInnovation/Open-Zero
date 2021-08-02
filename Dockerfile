##
## SETUP BUILDER
##

FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 as builder

## Install Prerequisites
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git wget unzip ninja-build python3 python3-pip gcc-8 g++-8 libeigen3-dev clang libopenblas-dev cmake && \
    pip3 install meson

# download and unpack stockfish
RUN cd ~ && \
    wget -4 https://stockfishchess.org/files/stockfish_14_linux_x64_avx2.zip -O stockfish.zip && \
    unzip stockfish.zip && \
    mv stockfish_*/stockfish_*_x64_avx2 stockfish

##
## SETUP RUNNER
##

FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
LABEL maintainer "PocInnovation <https://github.com/PocInnovation>"

# Updates
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget libopenblas-base python3 python3-pip && apt-get clean

# Copying data from current repository
WORKDIR /app
COPY . .

# Installing pip dependencies
RUN pip3 install -r requirements.txt

# Copy Stockfish executable
RUN mkdir /bins
COPY --from=builder /root/stockfish /bins/stockfish
ENV PATH /bins:$PATH

# run the tester here
RUN ldconfig
ENTRYPOINT [ "./launch-project.sh" ]
CMD [ "-m", "train" ]
