FROM python:3.10-slim

WORKDIR /app

COPY ./dictionary.py /app
COPY ./main.py /app
COPY ./slr.onnx /app
COPY ./requirements.txt /app

RUN pip install -r requirements.txt --no-cache-dir
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install libgl1


EXPOSE 30000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "30000"]