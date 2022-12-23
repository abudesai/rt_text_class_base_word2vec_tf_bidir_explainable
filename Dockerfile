FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get -y update && \
        apt-get -y install gcc mono-mcs && \
        apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt .
RUN pip install -r requirements.txt 

ENV embed_dim=100 
ENV embed_file_name=enwiki_20180420_"$embed_dim"d.txt

# For references please visit https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
RUN wget -P /tmp http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/"$embed_file_name".bz2 >> /tmp/download_stdout.txt
RUN bzip2 -d -q /tmp/"$embed_file_name".bz2

COPY app ./opt/app
WORKDIR /opt/app

# move embeddings into the location where model looks for it
RUN mv /tmp/"$embed_file_name" /opt/app/Utils/pretrained_embed
RUN echo "export embed_dim=${embed_dim}" >> /root/.bashrc #To keep env variable on the system after restarting
RUN echo "export embed_file_name=${embed_file_name}" >> /root/.bashrc #To keep env variable on the system after restarting


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"


RUN chmod +x train &&\
    chmod +x predict &&\
    chmod +x serve 

RUN chown -R 1000:1000 /opt/app/  && \
    chown -R 1000:1000 /var/log/nginx/  && \
    chown -R 1000:1000 /var/lib/nginx/

USER 1000