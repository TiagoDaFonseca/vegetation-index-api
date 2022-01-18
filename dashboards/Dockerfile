FROM python:3.9

WORKDIR /code/app/

ADD ./dashboards/module/* ./module/
COPY ./dashboards/hsi-dashboard.py ./hsi-dashboard.py
COPY ./install/dash-requirements.txt ./dash-requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache-dir -r ./dash-requirements.txt
RUN apt install net-tools -y

ENTRYPOINT ["streamlit", "run"]
CMD ["hsi-dashboard.py"]