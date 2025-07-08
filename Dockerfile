# Base Image
FROM python:3.8-slim-buster

WORKDIR /app/eprsa

RUN python -m pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy==1.24.4 biopython==1.81 fair-esm==2.0.0 pyyaml==6.0.1 transformers==4.31.0 huggingface-hub==0.16.4 && \
    apt-get -y update && \
    apt-get -y install vim

COPY . .

ENTRYPOINT ["/app/eprsa/e-prsa.py"]
