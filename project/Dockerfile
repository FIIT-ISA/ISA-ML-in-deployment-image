FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html  \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]