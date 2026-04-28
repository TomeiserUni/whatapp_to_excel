FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

RUN python -c "import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); import easyocr; easyocr.Reader(['pt'], gpu=False)"

RUN mkdir -p output

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/server.py"]
