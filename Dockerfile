FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential cmake git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT automatically — we read it in app.py
CMD ["python", "app.py"]

