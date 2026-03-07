FROM runpod/pytorch:2.1.0-py3.10-cuda11.8
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "handler.py"]
