#!/bin/zsh
# Re-runs the analysis steps in order. Data: ~/swot_p38_work (override with P38_DATA).
# Download steps (01, 04, 04b, 04d, 05, 05c, 06, 12a) are run separately if the data are not present.
# Usage: ./run_all.sh [first_step]   (steps: 07 05b 04c 07b 12b 12c 08 08a 08s 03 08d 08e 11 13 14 15)
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONWARNINGS=ignore
DATA=${P38_DATA:-~/swot_p38_work}
mkdir -p $DATA/results
LOG=$DATA/results/run_$(date +%Y%m%d_%H%M).log
STEPS=(07 05b 04c 07b 12b 12c 08 08a 08s 03 08d 08e 11 13 14 15)
FIRST=${1:-07}; started=0
for s in $STEPS; do
  [[ $s == $FIRST ]] && started=1
  (( started )) || continue
  echo "== $(date +%H:%M) step $s" >> $LOG
  case $s in
    07)  $PY 07_merge.py ;;
    05b) $PY 05b_gauge_location.py ;;
    04c) $PY 04c_climatology.py ;;
    07b) $PY 07b_sos_comparison.py ;;
    12b) $PY 12b_transfer_analysis.py ;;
    12c) $PY 12c_algorithm_coverage.py ;;
    08)  $PY 08_chain_inversion.py 10 ;;
    08a) $PY 08_chain_inversion.py 10 --ahg-min 0.05 --ahg-max 0.9 --tag _ahg005_09 ;;
    08s) $PY 08_chain_inversion.py 10 --sw-floor 0.15 --tag _sw015 ;;
    03)  $PY 03_twin_experiment_mc.py 200 10 ;;
    08d) $PY 08d_mcmc_diagnostics.py ;;
    08e) $PY 08e_mcmc_long.py ;;
    11)  $PY 11_summary.py ;;
    13)  $PY 13_figures_transfer.py ;;
    14)  $PY 14_figures_ablation.py ;;
    15)  $PY 15_tables.py ;;
  esac >> $LOG 2>&1 || { echo "!! step $s FAILED (code $?)" >> $LOG; exit 1; }
done
echo "== $(date +%H:%M) DONE" >> $LOG
