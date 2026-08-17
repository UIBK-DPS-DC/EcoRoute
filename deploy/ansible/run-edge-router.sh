#!/bin/bash

set -ex

export DEBIAN_FRONTEND=noninteractive

apt update -y
apt install -y nats-server

nats-server -c /etc/my-server.conf --name nats_edge -p 4222 -m 8222 &

python3 -m venv venv

source venv/bin/activate
cd code/router
pip install -r requirements.txt

export NATS_URL=localhost

uvicorn --host 0.0.0.0 --port 8000 router.app:app &

