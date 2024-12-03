FROM python:3.9-slim

WORKDIR /app
ENV no_proxy="localh
ENV https_proxy="http://DMZ
ENV http_proxy="http://DMZ-P

COPY ./requirements.txt /app/requirements.txt
COPY ./app /app/app

# Mettre à jour pip et setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN mkdir -p /app/logs

EXPOSE 8503

CMD ["streamlit", "run", "app/app_dmr.py", "--server.port=8503"]