import json
import sys

import great_expectations as gx
import pandas as pd


def run_validation(df: pd.DataFrame, name: str, expectations: list) -> dict:
    """Run a list of expectations on a DataFrame using a fresh ephemeral context."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas")
    asset = data_source.add_dataframe_asset(name=name)
    batch_def = asset.add_batch_definition_whole_dataframe("whole")

    suite = context.suites.add(gx.ExpectationSuite(name=f"{name}_suite"))
    for expectation in expectations:
        suite.add_expectation(expectation)

    validation = context.validation_definitions.add(
        gx.ValidationDefinition(data=batch_def, suite=suite, name=f"{name}_validation")
    )
    result = validation.run(batch_parameters={"dataframe": df})
    return {"success": result.success, "details": result.describe_dict()}


def finish(results: dict, report_path: str) -> None:
    """Write the report and exit with code 1 if any validation failed."""
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    failed = [name for name, r in results.items() if not r["success"]]
    if failed:
        print(f"Validation FAILED for: {failed}. See {report_path}")
        sys.exit(1)
    print("Validation passed.")
