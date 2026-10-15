import sys

import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def main(preprocessed_path: str, split_dir: str, report_path: str) -> None:
    # 1. Cargar parámetros desde params.yaml
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    train_split_target = params["split-data"]["train_split"]
    test_split_target = params["split-data"]["test_split"]

    # 2. Cargar datasets
    full = pd.read_csv(preprocessed_path)
    train = pd.read_csv(f"{split_dir}/train.csv")
    test = pd.read_csv(f"{split_dir}/test.csv")

    n_total = len(full)
    n_train = len(train)
    n_test = len(test)

    # 3. Calcular proporciones reales
    actual_train_ratio = n_train / n_total if n_total > 0 else 0
    actual_test_ratio = n_test / n_total if n_total > 0 else 0

    # 4. Definir una tolerancia estricta (ej. 1% de variación permitida por redondeos)
    tolerance = 0.01

    # 5. Validaciones personalizadas para los tamaños/proporciones
    train_proportional_success = abs(actual_train_ratio - train_split_target) <= tolerance
    test_proportional_success = abs(actual_test_ratio - test_split_target) <= tolerance

    results = {
        "train_proportion": {
            "success": train_proportional_success,
            "details": {
                "expected_ratio": train_split_target,
                "actual_ratio": actual_train_ratio,
                "expected_rows": round(n_total * train_split_target),
                "actual_rows": n_train
            }
        },
        "test_proportion": {
            "success": test_proportional_success,
            "details": {
                "expected_ratio": test_split_target,
                "actual_ratio": actual_test_ratio,
                "expected_rows": round(n_total * test_split_target),
                "actual_rows": n_test
            }
        }
    }

    # 6. Finalizar y generar reporte
    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])