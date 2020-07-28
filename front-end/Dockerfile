FROM nginx:latest

WORKDIR /usr/share/nginx/html

COPY . /usr/share/nginx/html

RUN apt-get update && apt-get install -qy vim && rm 50x.html


CMD ["nginx", "-g", "daemon off;"]
