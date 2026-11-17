FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CHECKPOINTS_DIR=/app/checkpoints \
    CONFIG_PATH=/app/config/mutimodal_docker.yaml \
    DEVICE=cuda


RUN test -f "$CONFIG_PATH" || (echo "ERROR: no se encontró $CONFIG_PATH" && exit 1)
RUN find "$CHECKPOINTS_DIR" -name "Best_training_state.pkl" | grep -q . \
    || (echo "ERROR: no se encontraron checkpoints en $CHECKPOINTS_DIR" && exit 1)

CMD ["python", "inferencia_docker.py"]