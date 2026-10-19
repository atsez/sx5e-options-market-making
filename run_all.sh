#!/usr/bin/env bash
# Reproduce every number and chart.
#   ./run_all.sh                                      synthetic test run
#   ./run_all.sh chain.csv 5512.3 history.csv real    run on a real chain
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -Wall -Wextra -o validate validate.cpp
g++ -O2 -std=c++17 -fopenmp -Wall -Wextra -o mmsim mmsim.cpp
./validate | tee validate_output.txt
python3 tests/test_surface.py
python3 tests/test_sim.py
if [ $# -ge 2 ]; then
  name="${4:-real}"
  hist_arg=()
  [ $# -ge 3 ] && hist_arg=(--history "$3")
  python3 pipeline.py --chain "$1" --spot "$2" "${hist_arg[@]}" --name "$name"
  python3 report.py "$name"
else
  python3 pipeline.py --synthetic
  python3 report.py synthetic
fi
