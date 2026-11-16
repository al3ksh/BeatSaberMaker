#!/bin/bash
# Runs the lines of queue.txt one by one (lines may be appended while it runs), then flushes Drive and stops the session.
# queue.txt line: <model> <run-name> <options...>   (model: timing, pattern, ornaments, layer-when, layer-what)
# Usage: run_queue.sh [RESULTS]   RESULTS = remote results folder; default gen2 (use .../checkpoints/layers for layer models)
# Needs once per session, by the user: BROWSER=echo colab drivemount -s train
# Test hooks: COLAB (colab binary), BSM_DIR (working folder), QUEUE_SLEEP (seconds between tries).
C="${COLAB:-$HOME/.local/bin/colab}"; cd "${BSM_DIR:-$HOME/bsm-colab}" || exit 1
D="--drive-dir /content/drive/MyDrive/BeatSaberMaker --work /content"
RESULTS="${1:-/content/drive/MyDrive/BeatSaberMaker/checkpoints/gen2}"
finished=0
finish() {  # flush Drive and stop the paid runtime; runs once on the normal path, again (harmlessly) from the trap
  [ "$finished" = 1 ] && return
  finished=1
  trap '' INT TERM HUP  # a second Ctrl-C must not skip colab stop
  printf '%s\n' 'from google.colab import drive' 'drive.flush_and_unmount()' 'print("drive flushed")' | "$C" exec -s train --timeout 900 >> queue.log 2>&1
  "$C" stop -s train >> queue.log 2>&1
  echo "$(date -u +%FT%TZ) QUEUE DONE ($1) $("$C" usage | head -1)" >> queue.log
}
trap 'finish "exit or signal"' EXIT
trap 'exit 130' INT TERM HUP
n=0
while :; do
  n=$((n+1)); line=$(sed -n "${n}p" queue.txt); [ -z "$line" ] && break
  set -- $line; model=$1; name=$2; shift 2
  echo "$(date -u +%FT%TZ) START $name ($model $*) $("$C" usage | head -1)" >> queue.log
  for try in 1 2 3; do
    "$C" exec -s train -f vm_train.py --env BSM_ARGS="--model $model --name $name $D -- $*" --timeout 0 > "run-$name.try$try.log" 2>&1
    code=$(sed 's/\x1b\[[0-9;]*m//g' "run-$name.try$try.log" | grep -o "vm_train: exit code [0-9]*" | tail -1 | grep -o "[0-9]*$")
    echo "$(date -u +%FT%TZ) END $name try $try exit ${code:-none}" >> queue.log
    [ "$code" = "0" ] && break
    [ "$try" -lt 3 ] && sleep "${QUEUE_SLEEP:-30}"
  done
  if [ "$code" != "0" ]; then echo "$(date -u +%FT%TZ) FAILED $name, queue stops" >> queue.log; break; fi
  mkdir -p "ckpt/$name"
  "$C" download -s train "$RESULTS/$name/best.pt" "ckpt/$name/best.pt" >> queue.log 2>&1
  "$C" download -s train "$RESULTS/$name/log.jsonl" "ckpt/$name/log.jsonl" >> queue.log 2>&1
done
finish "queue end"
