# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM ubuntu

RUN apt-get -y update && apt-get -y install nginx

COPY nginx.conf /etc/nginx/nginx.conf 
COPY index.html /usr/share/nginx/html
EXPOSE 8000/tcp

ENTRYPOINT ["nginx", "-g", "daemon off;"]
