$ErrorActionPreference = "Stop"

$PythonExe = "D:/anaconda/envs/deepnco_ml_312/python.exe"
$OutputRoot = "result/large_quantity_l9_fast_exact_repeat2000_20260615"

& $PythonExe experiments/run_large_algorithm_suite.py `
  --cases L9 `
  --algorithms tra_fast tra_exact `
  --tra-fast-iters 1 `
  --tra-fast-cap-sec 500 `
  --tra-fast-portfolio `
  --tra-fast-portfolio-candidates r3,g3 `
  --tra-fast-portfolio-workers 2 `
  --tra-exact-repeat-to-min-runtime `
  --tra-exact-min-runtime-sec 2000 `
  --tra-exact-max-repeat-attempts 16 `
  --tra-exact-repeat-orderings g3,r3,default `
  --tra-exact-iters 1 `
  --tra-exact-fix-time-sec 60 `
  --tra-exact-coarse-time-sec 10 `
  --tra-exact-timeout-sec 260 `
  --no-tra-exact-compiled-cache `
  --no-tra-exact-skip-initial-fixgurobi-eval `
  --tra-exact-allow-warm-start-fallback `
  --tra-exact-warm-start-subtask-ordering g3 `
  --tra-exact-use-seed-if-better `
  --tra-exact-layer-order Y `
  --tra-exact-revolving-mark-limit 4 `
  --tra-exact-candidate-stack-topk 2 `
  --tra-exact-max-candidate-stacks-per-order 8 `
  --tra-exact-candidate-station-topk-per-stack 1 `
  --tra-exact-seeded-fallback `
  --tra-exact-seed-candidates r3,g3 `
  --layered-sp4-mode greedy `
  --output-root $OutputRoot
