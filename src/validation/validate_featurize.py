# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals

import sys

import great_expectations.expectations as gxe
import numpy as np
import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def expected_one_hot(df: pd.DataFrame, col: str, prefix: str, categories: list) -> pd.DataFrame:
    """One-hot encode `col` with a fixed category list; unseen categories become all zeros."""
    dummies = pd.get_dummies(
        pd.Categorical(df[col], categories=categories), prefix=prefix, dtype=int
    )
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


def mismatched_columns(actual: pd.DataFrame, expected: pd.DataFrame, cols: list) -> list:
    """Columns whose values differ between actual and expected (NaN == NaN)."""
    bad = []
    for c in cols:
        a, e = actual[c], expected[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(e):
            same = np.isclose(a.to_numpy(float), e.to_numpy(float), equal_nan=True)
        else:
            same = ((a == e) | (a.isna() & e.isna())).to_numpy()
        if not same.all():
            bad.append(c)
    return bad


def check(success: bool, **details) -> dict:
    return {"success": bool(success), "details": details}


def check_imputation_statistics(df_in: pd.DataFrame, df_out: pd.DataFrame, numeric_cols: list, rel_tol: float = 0.1) -> dict:
    """
    Valida que la media, mediana y desviación estándar de las columnas numéricas 
    no varíen más allá de una tolerancia relativa (rel_tol) tras la imputación.
    """
    bad_cols = []
    metrics_details = {}

    for col in numeric_cols:
        if col not in df_in.columns or col not in df_out.columns:
            continue
        
        # Estadísticas previas a la imputación (pandas ignora NaN por defecto)
        mean_in = df_in[col].mean()
        median_in = df_in[col].median()
        std_in = df_in[col].std()

        # Estadísticas posteriores a la imputación
        mean_out = df_out[col].mean()
        median_out = df_out[col].median()
        std_out = df_out[col].std()

        # Calcular diferencia relativa (evitando división por cero)
        mean_diff = abs(mean_out - mean_in) / (abs(mean_in) + 1e-9)
        median_diff = abs(median_out - median_in) / (abs(median_in) + 1e-9)
        std_diff = abs(std_out - std_in) / (abs(std_in) + 1e-9)

        # Si alguna métrica supera la tolerancia permitida (ej. 10%)
        is_ok = (mean_diff <= rel_tol) and (median_diff <= rel_tol) and (std_diff <= rel_tol)
        
        if not is_ok:
            bad_cols.append(col)

        metrics_details[col] = {
            "mean_diff_pct": float(mean_diff * 100),
            "median_diff_pct": float(median_diff * 100),
            "std_diff_pct": float(std_diff * 100),
        }

    return {
        "success": len(bad_cols) == 0,
        "bad_columns": bad_cols,
        "metrics": metrics_details
    }


def validate_split(name, df_in, df_out, expected_oh, n_rows_in, dummy_cols, cat_col, is_train):
    """Build all validations for one split (train or test)."""
    numeric_cols = df_in.select_dtypes(include=np.number).columns.tolist()
    # Excluir columnas dummy del one-hot de las numéricas puras a evaluar por estadística
    numeric_cols = [c for c in numeric_cols if c not in dummy_cols and c != cat_col]

    expectations = [
        # Exact columns, in the exact order: original columns first, dummies last
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=expected_oh.columns.tolist()),
        # featurize.py never adds or removes rows
        gxe.ExpectTableRowCountToEqual(value=n_rows_in),
        *[gxe.ExpectColumnValuesToNotBeNull(column=c) for c in numeric_cols + dummy_cols],
        *[gxe.ExpectColumnValuesToBeInSet(column=c, value_set=[0, 1]) for c in dummy_cols],
    ]
    results = {name: run_validation(df_out, f"{name}_processed", expectations)}

    # Value-level checks solo si las formas coinciden
    if len(df_out) != len(expected_oh) or list(df_out.columns) != list(expected_oh.columns):
        results[f"{name}_values"] = check(
            False, reason="shape or columns differ from expected; see GX results"
        )
        return results

    # 1. Validación de Imputación por estabilidad estadística (Media, Mediana y DS)
    imputation_check = check_imputation_statistics(df_in, df_out, numeric_cols, rel_tol=0.15)
    results[f"{name}_imputation_stats"] = check(
        imputation_check["success"], 
        metrics=imputation_check["metrics"], 
        bad_columns=imputation_check["bad_columns"]
    )

    # 2. One-hot: values equal the expected encoding, and row sums are consistent[cite: 5]
    bad_oh = mismatched_columns(df_out, expected_oh, dummy_cols)
    row_sums = df_out[dummy_cols].sum(axis=1)
    rows_ok = (row_sums == 1).all() if is_train else (row_sums <= 1).all()
    results[f"{name}_one_hot"] = check(
        not bad_oh and rows_ok,
        mismatched_columns=bad_oh,
        rows_with_invalid_sum=int((row_sums != 1).sum() if is_train else (row_sums > 1).sum()),
    )

    # Infinite values check
    numeric_matrix = df_out.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    results[f"{name}_no_inf"] = check(not np.isinf(numeric_matrix).any())
    return results


def main(interim_dir: str, processed_dir: str, report_path: str) -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    f = params["featurize"]
    cat_col, prefix = f["categorical_col"], f["one_hot_prefix"]

    train_in = pd.read_csv(f"{interim_dir}/train.csv")
    test_in = pd.read_csv(f"{interim_dir}/test.csv")
    train_out = pd.read_csv(f"{processed_dir}/train.csv")
    test_out = pd.read_csv(f"{processed_dir}/test.csv")

    # Obtener categorías desde train_in para simular el one-hot esperado
    train_cat_series = train_in[cat_col].copy()
    categories = sorted(train_cat_series.dropna().unique())

    # Generar estructuras esperadas de one-hot sobre los datasets de entrada
    exp_train_oh = expected_one_hot(train_in, cat_col, prefix, categories)
    exp_test_oh = expected_one_hot(test_in, cat_col, prefix, categories)
    dummy_cols = [c for c in exp_train_oh.columns if c not in train_in.columns]

    results = {}
    results.update(
        validate_split(
            "train", train_in, train_out, exp_train_oh, len(train_in), dummy_cols, cat_col, is_train=True
        )
    )
    results.update(
        validate_split(
            "test", test_in, test_out, exp_test_oh, len(test_in), dummy_cols, cat_col, is_train=False
        )
    )
    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])