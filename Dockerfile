FROM  --platform=linux/amd64 python:3.8

WORKDIR /epicAi-container

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app ./app

RUN ls -la ./app

EXPOSE 10000

CMD ["python", "./app/main.py"]