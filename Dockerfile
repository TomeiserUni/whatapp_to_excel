FROM python:3.11-slim

# Dependências de sistema para EasyOCR e OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU-only primeiro (evita baixar versão CUDA de 4 GB)
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Pré-descarregar modelos EasyOCR durante o build (não no primeiro arranque)
RUN python -c "import easyocr; easyocr.Reader(['pt'], gpu=False)" 2>&1 | tail -5

RUN mkdir -p input output

VOLUME ["/app/input", "/app/output"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/pipeline.py"]
