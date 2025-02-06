FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 && \ 
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]