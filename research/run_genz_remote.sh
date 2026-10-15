#!/usr/bin/env bash
# run_genz_remote.sh — run the Genz benchmark on the supercomputer.
#
# Covers 7 architectures x {3-D, 6-D} x {canonical, tuned} = 28 jobs.
#
# Per AGENTS.md, each job runs in the background with stdout+stderr redirected
# to a log that mirrors the script name, inside the results mirror directory.
#
#   bash research/run_genz_remote.sh            # all 28 jobs
#   DIM=6 bash research/run_genz_remote.sh      # 6-D only
#   DIFF=tuned bash research/run_genz_remote.sh # tuned difficulty only
#   ARCHS="28 30" bash research/run_genz_remote.sh
#
# Results land in research/results/flat/<num>-<arch>-genz/<arch>_genz_n<n>_<tier>.csv

set -u

ARCHS="${ARCHS:-26 27 28 29 30 31 32}"
DIM="${DIM:-3 6}"
DIFF="${DIFF:-canonical tuned}"

for num in $ARCHS; do
  for n in $DIM; do
    for tier in $DIFF; do
      # map experiment number -> directory name and module
      case "$num" in
        26) slug="26-wire-genz";      mod="wire_genz"       ;;
        27) slug="27-siren-genz";     mod="siren_genz"      ;;
        28) slug="28-sechiren-genz";  mod="sechiren_genz"   ;;
        29) slug="29-tanhiren-genz";  mod="tanhiren_genz"   ;;
        30) slug="30-mixiren-genz";   mod="mixiren_genz"    ;;
        31) slug="31-cosiren-ps-genz"; mod="cosiren_ps_genz" ;;
        32) slug="32-gaussian-genz";  mod="gaussian_genz"   ;;
        *) echo "unknown experiment number: $num" >&2; continue ;;
      esac

      script="research/experiments/flat/${slug}/${mod}.py"
      logdir="research/results/flat/${slug}"
      log="${logdir}/${mod}_n${n}_${tier}.log"
      mkdir -p "$logdir"

      echo "launching  n=${n}  ${tier}  ${slug}"
      python "$script" --n "$n" --difficulty "$tier" &> "$log" &
    done
  done
done

echo
echo "all jobs launched; wait for completion, then check results with:"
echo "  python research/analysis/collect_results.py"
