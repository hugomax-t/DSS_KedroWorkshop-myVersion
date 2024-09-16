from kedro.pipeline import Pipeline, pipeline, node

from .nodes import(
    plot_confusion_matrix,
    plot_roc,
    plot_shap_feature_importance,
    plot_shap_summary
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            plot_confusion_matrix,
            inputs=["best_model_pipeline", "test_df"],
            outputs="confusion_matrix_plot",
            name="plot_confusion_matrix"
        ),
        node(
            plot_roc,
            inputs=["best_model_pipeline", "test_df"],
            outputs="ROC_plot",
            name="plot_ROC"
        ),
        node(
            plot_shap_summary,
            inputs=["best_model_pipeline", "test_df"],
            outputs="shap_summary_plot",
            name="plot_shap_summary"
        ),
        node(
            plot_shap_feature_importance,
            inputs=["best_model_pipeline", "test_df"],
            outputs="shap_feature_importance_plot",
            name="plot_shap_feature_importance"
        ),
    ])
