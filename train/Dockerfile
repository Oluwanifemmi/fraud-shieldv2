FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train/train.py train.py
COPY feature_store/feature_repo/fraud_features.py feature_store/feature_repo/fraud_features.py
COPY feature_store/feature_repo/feature_store.yaml feature_store/feature_repo/feature_store.yaml
COPY newdata/fraudTrain_sample.csv  newdata/fraudTrain_sample.csv

CMD ["python", "train.py"]