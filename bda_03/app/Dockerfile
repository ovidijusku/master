FROM python:3.10-slim

WORKDIR /
COPY requirements.txt /
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

CMD ["python", "app/insert.py"]