FROM openfaas/classic-watchdog:0.13.4 as watchdog
FROM ubuntu:16.04
RUN apt-get -q update -y && \
 apt-get -q install -y \
 apt-transport-https \
 apt-utils \
 build-essential \
 cron \
 curl \
 dsyslog \
 g++ \
 git \
 gcc \
 language-pack-id \
 libcairo2-dev \
 libexpat1 \
 libffi-dev \
 libssl-dev \
 libx11-dev \
 libxml2-dev \
 libxslt1-dev \
 libxss1 \
 nano \
 netcat \
 pango1.0-tests \
 psmisc \
 python \
 python-pip \
 python-dev \
 python-setuptools \
 ssl-cert \
 vim \
 zlib1g \
 zlib1g-dev
# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE
COPY — from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
# Add non root user
WORKDIR /home/app/
COPY index.py .
COPY requirements.txt .
ENV PATH=$PATH:/home/app/.local/bin:/home/app/python/bin/
ENV PYTHONPATH=$PYTHONPATH:/home/app/python:/home/app/function
RUN pip install -r requirements.txt
RUN mkdir -p function
RUN touch ./function/__init__.py
WORKDIR /home/app/function/
COPY function/requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /home/app/
COPY function function
ENV fprocess=”python index.py”
EXPOSE 8080
HEALTHCHECK — interval=3s CMD [ -e /tmp/.lock ] || exit 1
CMD [“fwatchdog”]
