FROM python:3.11-slim

WORKDIR /app

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY src/ ./src/
COPY data/prod.pkl    ./data/
COPY data/sku_map.pkl ./data/
COPY data/aliases.json ./data/

RUN mkdir -p output

ENV PYTHONUNBUFFERED=1
ENV RAILWAY_ENVIRONMENT=production

CMD ["python", "src/server.py"]
