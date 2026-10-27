#!/usr/bin/env bash
set -uo pipefail


echo "Running data collection"

mkdir -p results
sed -i 's|^\([[:space:]]*save_dir:[[:space:]]*\).*|\1./results|' vars/common.yml

rm group_vars/model_nvidia/secrets.yml

# Run your commands, ignoring failures
ansible-playbook -i inventory.ini debug_remote.yml --skip-tags "routers,nats" || true
ansible-playbook -i inventory.ini debug_models.yml --skip-tags models || true
python evaluation/eval.py http://router_edge -n 0.5 -t 5 -p ./prepared-datasets || true
sleep 120s || true
ansible-playbook -i inventory.ini metrics.yml -e "{\"queries_per_second\":0.5,\"duration\":10,\"lambda_u\":0.2,\"lambda_e\":0.5,\"lambda_t\":0.5}" || true



echo "Data collection completed."