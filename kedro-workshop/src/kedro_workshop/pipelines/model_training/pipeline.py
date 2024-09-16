from kedro.pipeline import Pipeline, pipeline, node

from .nodes import(
    split_training_data,
    train_model,
    reproduce_best_run
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            split_training_data,
            inputs=["clean_df", "params:test_size", "params:val_size", "params:random_state"],
            outputs=["X_train", "y_train", "X_val", "y_val", "test_df"],
            name="split_training_val_test_data"
        ),
        node(
            train_model,
            inputs=["X_train", "y_train", "X_val", "y_val", "params:cat_columns", "params:num_columns"],
            outputs="study",
            name="train_model"
        ),
        node(
            reproduce_best_run,
            inputs=["study", "X_train", "y_train", "X_val", "y_val", "params:cat_columns", "params:num_columns"],
            outputs="best_model_pipeline",
            name="reproduce_best_run"
        )
    ])
