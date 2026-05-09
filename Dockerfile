FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only first (avoids pulling ~4GB CUDA runtime)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and weights
COPY model.py .
COPY data.py .
COPY app.py .
COPY weights/mae_cifar10_150ep.pth weights/mae_cifar10_150ep.pth
COPY weights/probe.pth weights/probe.pth

# HF Spaces runs on port 7860 by default; override via docker-compose for local dev
ENV PORT=7860
EXPOSE ${PORT}

# Non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
USER appuser

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
