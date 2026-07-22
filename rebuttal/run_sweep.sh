#!/bin/bash
# Rank-sweep runner for the synthetic-control experiment (CLEAN datasets K1,K2,K4).
# For each config: train PILA, then run the test/analyzer step. Continues on failure.
# Progress -> rebuttal/logs/sweep_progress.log ; per-run logs -> rebuttal/logs/<name>_{train,test}.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p rebuttal/logs
PROG=rebuttal/logs/sweep_progress.log   # appended, so resumes keep their history

CONFIGS=$(ls rebuttal/configs/SYN_K*_r*.json | sort)
n=$(echo "$CONFIGS" | wc -l)
i=0
echo "[SWEEP START] $(date +%F_%H:%M:%S) — $n runs" >> "$PROG"
for cfg in $CONFIGS; do
  i=$((i+1))
  name=$(basename "$cfg" .json)
  # resumable: skip runs that already produced a test analyzer CSV
  if ls saved/mogi_synth/$name/*/models/model_best_testset_analyzer.csv >/dev/null 2>&1; then
    echo "[skip $i/$n] $name (already complete)" >> "$PROG"
    continue
  fi
  echo "[start $i/$n] $name $(date +%H:%M:%S)" >> "$PROG"
  if conda run -n mres python train_pila.py --config "$cfg" > "rebuttal/logs/${name}_train.log" 2>&1; then
    rundir=$(ls -td saved/mogi_synth/$name/*/ 2>/dev/null | head -1)
    if conda run -n mres python test_pila_mogi.py \
         --config "${rundir}models/config.json" \
         --resume "${rundir}models/model_best.pth" > "rebuttal/logs/${name}_test.log" 2>&1; then
      echo "[done $i/$n] $name $(date +%H:%M:%S) -> $rundir" >> "$PROG"
    else
      echo "[test-FAIL $i/$n] $name — see rebuttal/logs/${name}_test.log" >> "$PROG"
    fi
  else
    echo "[train-FAIL $i/$n] $name — see rebuttal/logs/${name}_train.log" >> "$PROG"
  fi
done
echo "[SWEEP DONE] $(date +%F_%H:%M:%S)" >> "$PROG"
