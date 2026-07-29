#!/bin/bash
# Multi-seed Austria runs for the Figure-8 variance question (reviewer Q2).
#
# Trains PILA and HVAE on the Austrian dataset at three seeds, then runs the
# test/analyzer step for each checkpoint so jm_seeds.py has its input CSVs.
#
# Run from anywhere:  ./rebuttal/seed_variance/run_seed_variance.sh
# Serial instead of parallel:  PARALLEL=0 ./rebuttal/seed_variance/run_seed_variance.sh
#
# All six runs are independent, so by default they are launched at once; this box
# has 256 cores and each run is capped at 8 threads. Wall clock ~2 h for the whole
# sweep in parallel (a single Austria run is ~2 h), ~12 h serial.
# Resumable: a run whose analyzer CSV already exists is skipped.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
PARALLEL="${PARALLEL:-1}"
mkdir -p "$HERE/logs"
PROG="$HERE/logs/progress.log"
SAVE_ROOT=saved/rebuttal/seed_variance

# Train, then test, one config. $1 = config path.
run_one() {
  local cfg="$1"
  local name; name=$(basename "$cfg" .json)
  local fam;  fam=${name%%_*}                       # PILA or HVAE
  local train_script test_script
  if [ "$fam" = "PILA" ]; then train_script=train_pila.py; test_script=test_pila_rtm.py
  else                          train_script=train_hvae.py; test_script=test_hvae_rtm.py; fi

  if ls $SAVE_ROOT/$name/*/models/model_best_testset_analyzer.csv >/dev/null 2>&1; then
    echo "[skip] $name (analyzer CSV already present)" >> "$PROG"; return 0
  fi

  echo "[start] $name $(date +%F_%H:%M:%S)" >> "$PROG"
  if ! $PYTHON "$train_script" --config "$cfg" > "$HERE/logs/${name}_train.log" 2>&1; then
    echo "[train-FAIL] $name -- see logs/${name}_train.log" >> "$PROG"; return 1
  fi
  local rundir; rundir=$(ls -td $SAVE_ROOT/$name/*/ 2>/dev/null | head -1)
  if ! $PYTHON "$test_script" --config "${rundir}models/config.json" \
                              --resume "${rundir}models/model_best.pth" \
                              > "$HERE/logs/${name}_test.log" 2>&1; then
    echo "[test-FAIL] $name -- see logs/${name}_test.log" >> "$PROG"; return 1
  fi
  echo "[done] $name $(date +%F_%H:%M:%S) -> $rundir" >> "$PROG"
}

CONFIGS=$(ls "$HERE"/configs/*_s*.json | sort)
echo "[SWEEP START] $(date +%F_%H:%M:%S) -- $(echo "$CONFIGS" | wc -l) runs, PARALLEL=$PARALLEL" >> "$PROG"
for cfg in $CONFIGS; do
  if [ "$PARALLEL" = "1" ]; then run_one "$cfg" & else run_one "$cfg"; fi
done
wait
echo "[SWEEP DONE] $(date +%F_%H:%M:%S)" >> "$PROG"
echo "Progress log: $PROG"
echo "Next:  $PYTHON rebuttal/seed_variance/jm_seeds.py"
