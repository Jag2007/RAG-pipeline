FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY rag ./rag
COPY static ./static

EXPOSE 5000
# one worker, because uploaded documents are kept in that worker's memory;
# threads let it serve the page while an answer is being generated
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180"]
