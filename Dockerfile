FROM python:3.11-slim

WORKDIR /app

COPY train/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY train/train.py train.py
COPY data/fraudTrain.csv data/fraudTrain.csv

CMD ["python", "train.py"]