#!/bin/bash
# PILA + physical-consistency study on GreenHyperSpectra (reviewer Q3).
#
# Trains PILA at four consistency-loss weights and scores each against the GHS
# measured traits, using the evaluator from rebuttal/greenhyperspectra/ (read-only).
#
#   ./rebuttal/consistency_loss/run_consistency_loss.sh              # all four, parallel
#   PARALLEL=0 ./rebuttal/consistency_loss/run_consistency_loss.sh   # one at a time
#   ONLY=S2 ./rebuttal/consistency_loss/run_consistency_loss.sh      # substring filter
#
# ~1.5 h per run (150 epochs, 1721 bands, CPU); the four are independent, so in
# parallel the sweep is ~1.5 h. Resumable: runs with metrics already written are
# skipped. Requires the GHS dataset built by
# rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
PARALLEL="${PARALLEL:-1}"
ONLY="${ONLY:-}"
mkdir -p "$HERE/logs" "$HERE/figures"
PROG="$HERE/logs/progress.log"
SAVE_ROOT=saved/rebuttal/consistency_loss
EVAL=rebuttal/greenhyperspectra/eval_greenhyperspectra.py

run_one() {
  local cfg="$1"
  local name; name=$(basename "$cfg" .json)

  if [ -f "$HERE/figures/${name}_metrics.json" ]; then
    echo "[skip] $name (metrics already written)" >> "$PROG"; return 0
  fi

  echo "[start] $name $(date +%F_%H:%M:%S)" >> "$PROG"
  if ! $PYTHON train_pila.py --config "$cfg" > "$HERE/logs/${name}_train.log" 2>&1; then
    echo "[train-FAIL] $name -- see logs/${name}_train.log" >> "$PROG"; return 1
  fi
  local rundir; rundir=$(ls -td $SAVE_ROOT/$name/*/ 2>/dev/null | head -1)
  if ! $PYTHON "$EVAL" --resume "${rundir}models/model_best.pth" \
                       --out_dir "$HERE/figures" \
                       > "$HERE/logs/${name}_eval.log" 2>&1; then
    echo "[eval-FAIL] $name -- see logs/${name}_eval.log" >> "$PROG"; return 1
  fi
  echo "[done] $name $(date +%F_%H:%M:%S) -> $rundir" >> "$PROG"
}

# _debug_*.json are 2-epoch smoke configs; never part of a sweep
CONFIGS=$(ls "$HERE"/configs/*.json | grep -v '/_debug_' | sort)
[ -n "$ONLY" ] && CONFIGS=$(echo "$CONFIGS" | grep -- "$ONLY")
[ -z "$CONFIGS" ] && { echo "no configs match ONLY='$ONLY'"; exit 1; }

echo "[SWEEP START] $(date +%F_%H:%M:%S) -- $(echo "$CONFIGS" | wc -l) runs, PARALLEL=$PARALLEL" >> "$PROG"
for cfg in $CONFIGS; do
  if [ "$PARALLEL" = "1" ]; then run_one "$cfg" & else run_one "$cfg"; fi
done
wait
echo "[SWEEP DONE] $(date +%F_%H:%M:%S)" >> "$PROG"
echo "Progress log: $PROG"
echo "Per-trait metrics: $HERE/figures/*_metrics.json"
