#!/bin/bash
# Screenshot one lab screen and check its DOM for crash messages.
# usage: tools/shoot.sh <lab-id> <hash> <out.png> [height]     e.g. tools/shoot.sh attention mask-3 /tmp/m3.png
# <hash> is <lessonId> or <lessonId>-<screen 1..4>. Pass "" as lab-id to shoot the gallery.
# Runs on an offline copy of the page (tools/offline.js), so headless Chrome never waits on the network.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB="$1"; HASH="$2"; OUT="$3"; HT="${4:-2000}"
F=$(node "$ROOT/tools/offline.js" "$LAB") || exit 1
URL="file://$F${HASH:+#$HASH}"
P=$(mktemp -d)
timeout 60 google-chrome --headless=new --disable-gpu --no-first-run --user-data-dir="$P" --window-size=1440,$HT --virtual-time-budget=4000 --screenshot="$OUT" "$URL" >/dev/null 2>&1
P2=$(mktemp -d)
timeout 60 google-chrome --headless=new --disable-gpu --no-first-run --user-data-dir="$P2" --window-size=1440,$HT --virtual-time-budget=4000 --dump-dom "$URL" 2>/dev/null > "$OUT.dom.html"
if [ ! -s "$OUT.dom.html" ]; then echo "TIMEOUT $LAB#$HASH (Chrome produced no DOM)";
elif grep -q 'class="err"' "$OUT.dom.html"; then echo "ERROR in $LAB#$HASH:"; sed -n 's/.*id="\(fatal-error\|lesson-error\)"[^>]*>\([^<]*\).*/\2/p' "$OUT.dom.html" | head -c 2000;
else echo "ok $LAB#$HASH ($(wc -c < "$OUT.dom.html") bytes dom)"; fi
rm -rf "$P" "$P2"
