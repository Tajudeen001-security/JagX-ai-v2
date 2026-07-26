FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential cmake git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install everything except llama-cpp-python normally
RUN grep -v llama-cpp-python requirements.txt > other-requirements.txt && \
    pip install --no-cache-dir -r other-requirements.txt

# Build llama-cpp-python from source WITHOUT AVX2/FMA (Render free CPU doesn't support them)
ENV CMAKE_ARGS="-DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_AVX_VNNI=OFF"
RUN pip install --no-cache-dir --force-reinstall --no-binary :all: llama-cpp-python==0.2.90

COPY . .

# Render sets $PORT automatically — we read it in app.py
CMD ["python", "app.py"]
