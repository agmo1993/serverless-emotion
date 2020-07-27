#!/usr/bin/env bash

echo "docker service create --replicas 1 -p 80:80 --name website website_image"

docker service create --replicas 1 -p 80:80 --name website website_image

