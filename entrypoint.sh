#!/bin/bash

mkdir -p /deploy/logs
touch /deploy/logs/error.log
touch /deploy/logs/access.log
cd /deploy

gunicorn APIMain:app -w 1 --threads 2 -b 0.0.0.0:8000
"$@"
