# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Copy Requirements ----------
COPY flask/ACE_bot/requirements.txt .

# ---------- Install Python Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Application Files ----------
COPY flask/ACE_bot/ .

# ---------- Environment Variables ----------
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

# ---------- Expose Port ----------
EXPOSE 5000

# ---------- Start Command ----------
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
