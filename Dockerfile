FROM python:3.10

RUN apt-get update \
 && apt-get install -y git \
 && apt-get clean

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "handler.py"]
