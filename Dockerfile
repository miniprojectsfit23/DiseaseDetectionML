FROM python:3.9.10-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
COPY . /code/
EXPOSE 8000
CMD [ "python", "./manage.py", "runserver","0.0.0.0:8000"]