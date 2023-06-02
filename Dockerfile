FROM python:3.10-bullseye

ARG ENV_VERSION=$ENV_VERSION
ENV VERSION=$ENV_VERSION
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONUNBUFFERED=1

# Legen Sie den Arbeitsverzeichnis fest
WORKDIR /app

# Kopieren Sie die Anwendung in den Container
COPY . .

RUN apt-get update 
# Installieren Sie die erforderlichen Abhängigkeiten
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

VOLUME /data
# Führen Sie das Skript aus
CMD ["python3","-u","./app.py"]
