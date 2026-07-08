# Match your CUDA 13.0 exactly
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10.12 (exact version you have)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /RSL_VG

COPY requirements.txt .

# Install pip packages using the NIGHTLY index for CUDA 13.0
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/nightly/cu130 \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0

# Now install your remaining requirements.txt (which should include faiss-gpu)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD CMD ["python", "inference_text2video.py", "--config_path", "configs/test_mimic.yaml"]