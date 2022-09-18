FROM python:3.7
WORKDIR /app

COPY src/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN apt update

RUN apt-get install -y libsndfile1

EXPOSE 8501

COPY src/. /app

ENTRYPOINT ["streamlit", "run"]

CMD ["Settings.py"]
