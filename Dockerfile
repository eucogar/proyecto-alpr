FROM python:3.11-slim


# Dependencias del sistema para OpenCV y EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
libgl1 \
libglib2.0-0 \
ffmpeg \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


COPY src /app/src
COPY data /app/data
COPY models /app/models


ENV PYTHONPATH=/app
CMD ["python", "-m", "src.main"]