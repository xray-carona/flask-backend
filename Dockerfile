FROM  python:3.6
MAINTAINER Ronald Das "ronald1das@gmail.com"
ENV PYTHONUNBUFFERED=1
RUN apt-get update -y
RUN  apt-get install -y libpq-dev wget unzip
RUN python -m pip install --upgrade pip
COPY requirements.txt /
RUN pip install -r requirements.txt
RUN pip install awscli
COPY . /app
WORKDIR /app
ARG MODEL_CKPT
ARG MODEL_META
ARG MODEL_CHESTER
ARG MODEL_UNETCT
RUN aws s3 cp $MODEL_CHESTER /app/model/ChesterAI/
RUN aws s3 cp $MODEL_CKPT /app/model/COVID-Netv1/
RUN aws s3 cp $MODEL_META /app/model/COVID-Netv1/
RUN aws s3 cp $MODEL_UNETCT /app/model/U-Net-CT/
#RUN wget $MODEL_CKPT
#RUN cp model.data-00000-of-00001 /app/model/COVID-Netv1/model.data-00000-of-00001
#RUN wget $MODEL_META
#RUN cp model.meta /app/model/COVID-Netv1/model.meta
RUN file="$(ls /app/model/COVID-Netv1/)" && echo $file
EXPOSE 5000
CMD ["gunicorn"  , "--bind", "0.0.0.0:5000","--timeout","300", "wsgi:app","--access-logfile","'-'"]
