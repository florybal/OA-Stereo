FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    ffmpeg \
    libsm6 \
    libxext6

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
        PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME

COPY --chown=user . $HOME

RUN git clone https://github.com/c237814486/OA-Stereo

RUN pip install --no-cache-dir -r requirements.txt
