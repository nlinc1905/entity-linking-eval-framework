FROM python:3.10-slim-bullseye

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["gunicorn", "--workers=1", "--threads=1", "-b 0.0.0.0:80", "app:server"]
