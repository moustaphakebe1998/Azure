# Docker file
FROM python:3.9-slim

WORKDIR /app
ENV no_proxy="localhost,1
ENV https_proxy="http:
ENV http_proxy="http://

COPY ./requirements.txt /app/requirements.txt
COPY ./app /app/app
COPY ./modele_langdetect /app/modele_langdetect

# Mettre à jour pip et setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Installer les dépendances système requises pour la compilation
RUN apt-get update && \
    apt-get install --no-install-recommends --yes build-essential && \
    rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN mkdir -p /app/logs

EXPOSE 8506

CMD ["streamlit", "run", "app/app_dmr.py", "--server.port=8506"]