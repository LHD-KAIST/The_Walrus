
"""파라미터 스윕을 수행해 GBS 분류 파이프라인의 정확도를 기록하는 스크립트."""
from __future__ import annotations

import csv
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable

from GBS_image import ExperimentConfig, GBSPipeline

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


PARAM_GRID = {
    "modes": [2, 3, 4, 5],
    "cutoff": [2, 3, 4],
    "shots_per_sample": [5, 6, 7],
    "max_squeezing": [0.20, 0.25, 0.30, 0.35],
    "max_train_samples": [240],
    "max_test_samples": [120],
    "perceptron_max_iter": [1500, 2000],
    "gelm_hidden_units": [48, 64],
    "gelm_reg": [1e-3],
    "grvfl_hidden_units": [32, 40],
    "grvfl_reg": [1e-3],
}


def sweep(param_grid: Dict[str, Iterable]) -> None:
    keys = list(param_grid.keys())
    values_product = list(product(*[param_grid[k] for k in keys]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DATA_DIR / f"gbs_sweep_{timestamp}.csv"

    fieldnames = [
        *keys,
        "perceptron_acc",
        "gelm_acc",
        "grvfl_acc",
        "status",
        "message",
    ]

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for combination in values_product:
            params = dict(zip(keys, combination))
            config = ExperimentConfig()

            config.gbs.modes = params["modes"]
            config.gbs.cutoff = params["cutoff"]
            config.gbs.shots_per_sample = params["shots_per_sample"]
            config.gbs.max_squeezing = params["max_squeezing"]
            config.gbs.interferometer_seed = 42 + params["modes"]

            config.max_train_samples = params["max_train_samples"]
            config.max_test_samples = params["max_test_samples"]
            config.perceptron_max_iter = params["perceptron_max_iter"]

            config.gelm.hidden_units = params["gelm_hidden_units"]
            config.gelm.reg = params["gelm_reg"]

            config.grvfl.hidden_units = params["grvfl_hidden_units"]
            config.grvfl.reg = params["grvfl_reg"]

            print(
                (
                    "[스윕] modes={modes}, cutoff={cutoff}, shots={shots}, max_sq={max_sq:.2f}, "
                    "gelm_hidden={gelm_hidden}, grvfl_hidden={grvfl_hidden}"
                ).format(
                    modes=params["modes"],
                    cutoff=params["cutoff"],
                    shots=params["shots_per_sample"],
                    max_sq=params["max_squeezing"],
                    gelm_hidden=params["gelm_hidden_units"],
                    grvfl_hidden=params["grvfl_hidden_units"],
                )
            )

            pipeline = GBSPipeline(config)

            row: Dict[str, object] = {
                **params,
                "perceptron_acc": None,
                "gelm_acc": None,
                "grvfl_acc": None,
                "status": "ok",
                "message": "",
            }

            try:
                metrics = pipeline.evaluate(verbose=False)
            except MemoryError as exc:
                row["status"] = "memory_error"
                row["message"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error"
                row["message"] = str(exc)
            else:
                row.update(
                    perceptron_acc=metrics["perceptron"],
                    gelm_acc=metrics["gelm"],
                    grvfl_acc=metrics["grvfl"],
                )

            writer.writerow(row)

    print(f"결과를 {output_path}에 저장했습니다.")


def main() -> None:
    sweep(PARAM_GRID)


if __name__ == "__main__":
    main()
