FROM continuumio/miniconda3

LABEL MAINTAINER="Doaa Altarawy <doaa.altarawy@gmail.com>"


ENV GROUP_ID=1000 \
    USER_ID=1000

WORKDIR /var/www/

ADD ./requirements.txt /var/www/requirements.txt

# what's only needed for continuumio/miniconda3
RUN apt-get update && \
    pip install --upgrade pip setuptools wheel && \
    conda install bottleneck && \
    apt-get install -y --no-install-recommends npm && \
    rm -rf /var/lib/apt/lists/*  && \
    # Python
    pip install --no-cache-dir -r requirements.txt && \
    conda clean -y --all


ADD . /var/www/

# Install Javascript requirements (from package.json file)
RUN cd app/static && \
    npm install && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g $GROUP_ID www  && \
    useradd -r -g www -s /bin/sh -u $USER_ID www  && \
    chown -R www:www /var/www

USER www


# Listens on port, doesn't publish it. Publish by -p or ports in docker-compose
EXPOSE 5000

# Run in Exec form, can't be overriden
ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:5000", "covid_xrays:app"]
# Params to pass to ENTRYPOINT, and can be overriden when running containers
CMD ["-w", "2", "--access-logfile", "access.log", "--error-logfile", "error.log"]

