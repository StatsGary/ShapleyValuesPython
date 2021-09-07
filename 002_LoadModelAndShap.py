# =============================================================================
# Script Purpose:          Load model and look at feature importance with SHAP
# Author:                  Gary Hutson @ hutsons-hacks.info
# Date:                    07/09/2021
# =============================================================================
import shap
from pickle import load
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
# Load in pickle file
# =============================================================================
filename = "stranded_model.sav"
# Previous pickle file returned a list so we will perform 
# multiple assignment here
model, X_train, X_test, Y_train, Y_test = load(open(filename, 'rb')) 


# =============================================================================
# Make model predictions
# =============================================================================
# Make predictions with model
pred_class = model.predict(X_test)
pred_probs = model.predict_proba(X_test)

# =============================================================================
# Evaluate the model 
# =============================================================================

def confusion_matrix_eval(Y_truth, Y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(Y_test, pred_class)
    cr = classification_report(Y_test, pred_class)
    print("-"*90)
    print("[CLASS_REPORT] printing classification report to console")
    print(cr)
    print("-"*90)
    return [cm, cr]

cm = confusion_matrix_eval(Y_test, pred_class)



# =============================================================================
# Shapley Values for Feature Importance
# =============================================================================
# Fit relevant explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)


# View shap values

print(shap_values)

# =============================================================================
# # Get global variable importance plot
# =============================================================================
plt_shap = shap.summary_plot(shap_values, features=X_train, 
                             feature_names=X_train.columns, 
                             show=False,
                             plot_size=(30,15))
plt.savefig("global_shap.png")


# =============================================================================
# # Local Interpretation Plots
# =============================================================================
obs_idx = 488
local_plot = shap.force_plot(explainer.expected_value, shap_values[obs_idx], 
                 features=X_train.loc[obs_idx],
                 feature_names=X_train.columns,
                 show=False, matplotlib=True)
plt.savefig("force_plot.png")

shap.plots.bar(shap_values.abs, color="blue")

