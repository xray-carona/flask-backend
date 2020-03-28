FROM tensorflow/tensorflow:1.15.2-py3
MAINTAINER Ronald Das "ronald1das@gmail.com"
ENV PYTHONUNBUFFERED=1
RUN apt-get update -y
#RUN apt-get install -y git libmysqlclient-dev libxml2-dev unzip wget
#RUN apt-get install -y python3-pip python3-dev build-essential libmysqlclient-dev libxml2-dev
COPY requirements.txt /
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
RUN wget file_loc
RUN cp model.ckpt-0.data-00000-of-00001?dl=0 /app/models/model.ckpt-0.data-00000-of-00001
RUN file="$(ls /app/models/)" && echo $file
EXPOSE 5006
#ENTRYPOINT ["python"]
#CMD ["home.py"]
CMD ["gunicorn"  , "--bind", "0.0.0.0:5006","--timeout","300", "home:app","--access-logfile","'-'"]