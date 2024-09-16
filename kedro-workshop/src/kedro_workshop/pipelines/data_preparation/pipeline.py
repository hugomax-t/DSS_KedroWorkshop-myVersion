from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    process_cabin,
    fill_missing_values,
    drop_columns
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            process_cabin,
            inputs="train_df",
            outputs="processed_cabin",
            name="process_cabin_column"
        ),
        node(
            fill_missing_values,
            inputs=["processed_cabin", "params:fill_na_strategy"],
            outputs="filled_missing_values",
            name="fill_missing_values"
        ),
        node(
            drop_columns,
            inputs=["filled_missing_values", "params:cols_to_drop"],
            outputs="clean_df",
            name="drop_unused_columns"
        )
    ])
