FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
LABEL maintainer "PocInnovation <https://github.com/PocInnovation>"

## Install Prerequisites
RUN apt-get update && \
 DEBIAN_FRONTEND=noninteractive apt-get install -y wget wget unzip \
 python3 python3-pip && apt-get clean

# download and unpack stockfish
RUN cd ~ && \
    wget -4 https://stockfishchess.org/files/stockfish_14_linux_x64_avx2.zip -O stockfish.zip && \
    unzip stockfish.zip && \
    mv stockfish_*/stockfish_*_x64_avx2 stockfish

# Copying data from current repository
WORKDIR /app
COPY . .

# Installing pip dependencies
RUN pip3 install -r requirements.txt

# Copy Stockfish executable
RUN mkdir /bins
RUN cd ~ && mv stockfish /bins/. && cd -
ENV PATH /bins:$PATH

# run the tester here
ENTRYPOINT [ "./launch-project.sh" ]
CMD [ "-m", "train" ]
