# Basis-Image mit Python
FROM python:3.11-slim

# Logs lesbar + keine pyc-Dateien
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Arbeitsverzeichnis im Container
WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projekt kopieren
COPY ml_playground ./ml_playground

# Streamlit-Port
EXPOSE 8501

# Start-Kommando
CMD ["streamlit", "run", "ml_playground/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
