FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY src/ ./src/
# Catálogo vem da Shopkit em runtime (sem prod.pkl/sku_map.pkl na imagem).
# Só os aliases curados e exemplos RAG, que são pequenos e úteis ao matching.
COPY data/aliases.json  ./data/
COPY data/exemplos.json ./data/

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/server.py"]
