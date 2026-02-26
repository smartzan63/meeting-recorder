FROM python:3.11-slim

# ffmpeg is the only native dep — used for audio conversion before Gemini upload
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pipeline.py obs.py config.py ./
COPY static/ ./static/

RUN mkdir -p recordings transcripts

ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
