#!/bin/sh

nginx
certbot --nginx --non-interactive --agree-tos -m doaa.altarawy@gmail.com -d myapp.azure.com
while true; do sleep 1; done