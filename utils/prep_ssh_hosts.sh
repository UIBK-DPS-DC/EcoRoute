#!/usr/bin/env bash
set -euo pipefail

INVENTORY="../deploy/ansible/inventory.ini"
KNOWN_HOSTS="$HOME/.ssh/known_hosts"

# Extract all IPs from ansible_host= fields
IPS=$(grep -oP 'ansible_host=\K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$INVENTORY" | sort -u)

echo "Found IPs:"
echo "$IPS"
echo

for ip in $IPS; do
    echo "Processing $ip"

    # Remove old entry (ignore errors if not present)
    ssh-keygen -f "$KNOWN_HOSTS" -R "$ip" 2>/dev/null || true

    # Preload host key so SSH won't prompt
    ssh-keyscan -H "$ip" >> "$KNOWN_HOSTS" 2>/dev/null

done

# Deduplicate known_hosts safely
sort -u "$KNOWN_HOSTS" -o "$KNOWN_HOSTS"

echo "Done. SSH connections should no longer prompt for host key confirmation."