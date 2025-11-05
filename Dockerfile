FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Opción 1 (si ya pusiste la condición de plataforma en requirements.txt):
RUN pip install --no-cache-dir -r requirements.txt

# ==== Paquetes CRISP-DM publicados en PyPI ====
RUN pip install --no-cache-dir \
    evaluation-stage \
    data-understanding \
    modeling-stage \
    business-understanding \
    deployment-stage \
    data-preparation-ml

# Copiamos el script que está en CRISP-DM/
COPY CRISP-DM/run_pipelinePyPI.py ./run_pipelinePyPI.py

RUN mkdir -p artifacts

CMD ["python", "run_pipelinePyPI.py"]
