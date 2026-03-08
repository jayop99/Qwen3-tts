FROM python:3.10

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "handler.py"]
