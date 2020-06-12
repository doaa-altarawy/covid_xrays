FROM alpine:latest

LABEL MAINTAINER="Doaa Altarawy <doaa.altarawy@gmail.com>"

RUN apk --update add nginx && \
    ln -sf /dev/stdout /var/log/nginx/access.log && \
    ln -sf /dev/stderr /var/log/nginx/error.log && \
    mkdir /etc/nginx/sites-enabled/ && \
    mkdir -p /run/nginx && \
    rm -rf /etc/nginx/conf.d/default.conf && \
    rm -rf /var/cache/apk/*

COPY conf.d/app.conf /etc/nginx/conf.d/app.conf
COPY boot.sh boot.sh
RUN chmod +x boot.sh

EXPOSE 80 443

RUN apk add --no-cache certbot certbot-nginx && \
    echo "0 0,12 * * * root python -c 'import random; import time; time.sleep(random.random() * 3600)' && certbot renew -q" | tee -a /etc/crontab > /dev/null

RUN apk add --no-cache openrc busybox-initscripts && \
    rc-update add crond

CMD "./boot.sh"