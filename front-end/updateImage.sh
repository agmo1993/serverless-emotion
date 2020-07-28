#!/usr/bin/env bash

docker build -t website_image .

docker service update --image website_image website --force
