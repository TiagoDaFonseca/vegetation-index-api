FROM python:3.9

WORKDIR /code

ADD ./app/api ./api
COPY ./install/api-requirements.txt ./api-requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache-dir -r ./api-requirements.txt


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]