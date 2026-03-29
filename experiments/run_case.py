from experiments.run_benchmark import run_small_layer_augmented_case_export

result_root = run_small_layer_augmented_case_export(
    seed=42,
    max_iters=200,
    no_improve_limit=100,
    epsilon=0.05,
    sp2_time_limit_sec=10.0,
    sp4_lkh_time_limit_seconds=5,
    export_best_solution=False,
    case='SMALL',
    silent=True,
)
print(result_root)
