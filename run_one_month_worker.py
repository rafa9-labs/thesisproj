import os
import json
import argparse
import numpy as np

# Silence TF logs in worker
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Don't let TF pre-allocate all VRAM
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to run_config.worker.json")
    ap.add_argument("--out_dir", required=True, help="Month output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 11111))
    _seed_everything(seed)

    month_idx = int(cfg["month_idx"])
    starting_equity = float(cfg.get("starting_equity", 1000.0))
    starting_position = int(cfg.get("starting_position", 0))

    init_kwargs = cfg.get("init_kwargs", {})

    # Import AFTER TF config/seed
    # ADAPT HERE if your class name / file differs:
    from MLBacktesterNoWFO import MLBacktester

    bt = MLBacktester(**init_kwargs)

    res = bt.run_single_month(
        month_idx=month_idx,
        out_dir=args.out_dir,
        seed=seed,
        starting_equity=starting_equity,
        starting_position=starting_position,
    )

    # Save scalar summary
    out_json = os.path.join(args.out_dir, "month_result.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    # Save bar-wise results if available
    df = getattr(bt, "results", None)
    if df is not None:
        out_csv = os.path.join(args.out_dir, "month_bars.csv")
        try:
            df.to_csv(out_csv, index=True)
        except Exception:
            # If df isn't a pandas DF for some reason, ignore
            pass

if __name__ == "__main__":
    main()