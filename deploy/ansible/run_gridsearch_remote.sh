#!/usr/bin/env bash
set -uo pipefail

CONFIG="config/router/config.yml"

# Define the possible values
u_values=(0.05 0.1 0.2)
e_values=(0.1 0.3 0.5)
t_values=(0.25 0.5 0.75)

for u in "${u_values[@]}"; do
    for e in "${e_values[@]}"; do
        for t in "${t_values[@]}"; do
            echo "Running combination: u=$u, e=$e, t=$t"

            # Update the configuration
            sed -i "/^trainer:/,/^[^[:space:]#]/ {
                s/^\([[:space:]]*lambda_utilization:[[:space:]]*\).*/\1$u/
                s/^\([[:space:]]*lambda_efficiency:[[:space:]]*\).*/\1$e/
                s/^\([[:space:]]*lambda_time:[[:space:]]*\).*/\1$t/
            }" "$CONFIG"

            # Run your commands, ignoring failures
            ansible-playbook -i inventory.ini debug_remote.yml --skip-tags "routers,nats" --vault-password-file ~/.vault_pass.txt || true
            ansible-playbook -i inventory.ini debug_models.yml --skip-tags models --vault-password-file ~/.vault_pass.txt || true
            python evaluation/eval.py http://router_edge -n 0.5 -t 10 -p ../edge-cloud-llm-routing/src/hf_tmp/targets || true
            sleep 120s || true
            curl http://router_edge:8000/save || true
            curl http://router_uc:8000/save || true
            curl http://router_tacc:8000/save || true
            ansible-playbook -i inventory.ini metrics.yml -e "{\"queries_per_second\":0.5,\"duration\":10,\"lambda_u\":$u,\"lambda_e\":$e,\"lambda_t\":$t}" --vault-password-file ~/.vault_pass.txt || true

            echo "Finished combination: u=$u, e=$e, t=$t"
            echo "----------------------------------------"
        done
    done
done

echo "All combinations completed."