import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn import set_config
from sklearn.pipeline import Pipeline

# Set transformer output to be a pandas DataFrame instead of numpy array
set_config(transform_output = "pandas")

import matplotlib.pyplot as plt
import numpy as np
import shap

def plot_confusion_matrix(model_pipeline: Pipeline, test_df: pd.DataFrame) -> plt.figure:
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    model_pipeline (Pipeline): Trained model pipeline.
    test_df (pd.DataFrame): DataFrame containing test data.

    Returns:
    plt.figure: Confusion matrix plot.
    """

    X_test = test_df.drop(["Transported"], axis=1)
    y_test = test_df["Transported"]

    y_pred = model_pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, normalize = 'true')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Not Transported', 'Transported']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(np.round(cm[i, j], 2)),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    confusion_matrix_plot = plt.gcf()

    return confusion_matrix_plot

def plot_roc(model_pipeline: Pipeline, test_df: pd.DataFrame) -> plt.figure:
    """
    Plot the ROC curve for the model predictions.

    Parameters:
    model_pipeline (Pipeline): Trained model pipeline.
    test_df (pd.DataFrame): DataFrame containing test data.

    Returns:
    plt.figure: ROC curve plot.
    """
    X_test = test_df.drop(["Transported"], axis=1)
    y_test = test_df["Transported"]
    y_pred_probs = model_pipeline.predict_proba(X_test)[:,1]

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_pred_probs)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    roc_auc_plot = plt.gcf()

    return roc_auc_plot

def plot_shap_summary(model_pipeline: Pipeline, test_df: pd.DataFrame) -> plt.figure:
    """
    Plot the SHAP summary plot for feature importance and partial dependence.

    Parameters:
    model_pipeline (Pipeline): Trained model pipeline.
    test_df (pd.DataFrame): DataFrame containing test data.

    Returns:
    plt.figure: SHAP summary plot.
    """
    X_test = test_df.drop(["Transported"], axis=1)
    y_test = test_df["Transported"]

    model = model_pipeline['model']

    X_test = model_pipeline['feature_engineering'].transform(X_test)    

    # Create a SHAP Tree Explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the chosen sample
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, show=False, plot_size = 0.2)

    summary_plot = plt.gcf()
    
    return summary_plot

def plot_shap_feature_importance(model_pipeline: Pipeline, test_df: pd.DataFrame) -> plt.figure:
    """
    Plot the SHAP feature importance bar plot.

    Parameters:
    model_pipeline (Pipeline): Trained model pipeline.
    test_df (pd.DataFrame): DataFrame containing test data.

    Returns:
    plt.figure: SHAP feature importance plot.
    """
    X_test = test_df.drop(["Transported"], axis=1)
    y_test = test_df["Transported"]

    model = model_pipeline['model']

    X_test = model_pipeline['feature_engineering'].transform(X_test)    

    # Create a SHAP Tree Explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the chosen sample
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, plot_size = 0.2)

    feature_importance_plot = plt.gcf()
    
    return feature_importance_plot