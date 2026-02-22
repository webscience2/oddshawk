#!/usr/bin/env bash
# Bootstrap oddshawk secrets from Bitwarden.
# Prerequisites: bw CLI installed and logged in (bw login / bw unlock)
#
# Stores secrets as a Bitwarden secure note named "oddshawk".
# The note has custom fields for each secret, plus file attachments
# for the Betfair SSL cert/key.
#
# Usage:
#   ./setup.sh          # pull secrets and write .env + certs
#   ./setup.sh push     # save current .env + certs TO Bitwarden

set -euo pipefail

ITEM_NAME="oddshawk"

# Ensure bw is available and unlocked
if ! command -v bw &>/dev/null; then
    echo "Error: Bitwarden CLI (bw) not found."
    echo "Install: brew install bitwarden-cli"
    exit 1
fi

# Check session
if ! bw status 2>/dev/null | grep -q '"unlocked"'; then
    echo "Bitwarden vault is locked. Run: export BW_SESSION=\$(bw unlock --raw)"
    exit 1
fi

bw sync --quiet 2>/dev/null || true

# Get the item ID
ITEM_ID=$(bw list items --search "$ITEM_NAME" 2>/dev/null \
    | python3 -c "import sys,json; items=json.load(sys.stdin); print(items[0]['id'] if items else '')" 2>/dev/null)

if [ "${1:-}" = "push" ]; then
    # ── Push current secrets TO Bitwarden ──
    if [ ! -f .env ]; then
        echo "Error: no .env file found"
        exit 1
    fi

    ENV_CONTENT=$(cat .env)

    if [ -z "$ITEM_ID" ]; then
        echo "Creating new Bitwarden item '$ITEM_NAME'..."
        TEMPLATE=$(bw get template item)
        NEW_ITEM=$(echo "$TEMPLATE" | python3 -c "
import sys, json
item = json.load(sys.stdin)
item['name'] = '$ITEM_NAME'
item['type'] = 2  # secure note
item['secureNote'] = {'type': 0}
item['notes'] = '''$ENV_CONTENT'''
json.dump(item, sys.stdout)
")
        ITEM_ID=$(echo "$NEW_ITEM" | bw encode | bw create item | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
        echo "Created item: $ITEM_ID"
    else
        echo "Updating existing Bitwarden item '$ITEM_NAME'..."
        bw get item "$ITEM_ID" \
            | python3 -c "
import sys, json
item = json.load(sys.stdin)
item['notes'] = '''$ENV_CONTENT'''
json.dump(item, sys.stdout)
" \
            | bw encode | bw edit item "$ITEM_ID" >/dev/null
        echo "Updated item."
    fi

    # Attach certs if they exist
    for f in betfair-client.crt betfair-client.key; do
        if [ -f "$f" ]; then
            # Delete existing attachment with same name (if any)
            ATTACH_ID=$(bw get item "$ITEM_ID" 2>/dev/null \
                | python3 -c "
import sys, json
item = json.load(sys.stdin)
for a in item.get('attachments', []):
    if a['fileName'] == '$f':
        print(a['id'])
        break
" 2>/dev/null || true)
            if [ -n "$ATTACH_ID" ]; then
                bw delete attachment "$ATTACH_ID" --itemid "$ITEM_ID" 2>/dev/null || true
            fi
            bw create attachment --file "$f" --itemid "$ITEM_ID" >/dev/null
            echo "Attached $f"
        fi
    done

    echo "Done. Secrets pushed to Bitwarden."

elif [ -z "${1:-}" ]; then
    # ── Pull secrets FROM Bitwarden ──
    if [ -z "$ITEM_ID" ]; then
        echo "Error: no Bitwarden item named '$ITEM_NAME' found."
        echo "Run './setup.sh push' first to store your secrets."
        exit 1
    fi

    echo "Pulling secrets from Bitwarden..."

    # Get .env content from notes field
    bw get item "$ITEM_ID" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('notes',''))" \
        > .env
    echo "Wrote .env"

    # Download cert attachments
    ATTACHMENTS=$(bw get item "$ITEM_ID" \
        | python3 -c "
import sys, json
item = json.load(sys.stdin)
for a in item.get('attachments', []):
    print(a['id'], a['fileName'])
" 2>/dev/null || true)

    while IFS=' ' read -r aid aname; do
        if [ -n "$aid" ]; then
            bw get attachment "$aid" --itemid "$ITEM_ID" --output "$aname" >/dev/null
            echo "Wrote $aname"
        fi
    done <<< "$ATTACHMENTS"

    echo "Done. Run: uv run python3 main.py"

else
    echo "Usage: ./setup.sh [push]"
    exit 1
fi
