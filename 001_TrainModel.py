# =============================================================================
# Script Purpose:          Training Model Step
# Author:                  Gary Hutson @ hutsons-hacks.info
# Date:                    07/09/2021
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer


# =============================================================================
# Data reading and cleansing
# =============================================================================
# Read in data
df = pd.read_csv("https://hutsons-hacks.info/wp-content/uploads/2021/04/LongLOSData.csv")
df = df.dropna()

# Encode categorical variables in X data frame
Y = df["stranded.label"]

# Select df columns for X
drop_cols = ["stranded.label", "admit_date"]
X = df.loc[:, [x for x in df.columns if x not in drop_cols]]

# Dummy encode the categorical labels
X = pd.get_dummies(X, columns=['frailty_index'])

# Drop one reference column as it does not give us much information and could
# cause multicollinearity affects in linear models
X = X.drop(["frailty_index_No index item"], axis=1)

# =============================================================================
# Feature Selection using Recursive Feature Engineering
# =============================================================================

# Feature selection using RFE
def recursive_feature_eng(model, X, Y):
    print("[INFO] Starting Recursive Feature Engineering")
    rfe = RFE(model)
    rfe_fit = rfe.fit(X,Y)
    print("Number of features chosen: %d" % rfe_fit.n_features_)
    print("Selected features chosen: %s" % rfe_fit.support_)
    print("Fit ranking of feature importance: %s" % rfe_fit.ranking_)
    print("[INFO] Ending Recursive Feature Engineering")
    return [rfe_fit, rfe_fit.n_features_, rfe_fit.support_]
    

rfe_model = LogisticRegression(solver='liblinear')
rfe_fit = recursive_feature_eng(rfe_model, X, Y)

# Pull out the feature ranking from the fitted object
columns_to_remove = rfe_fit[2]
X_reduced = X.loc[:,columns_to_remove]

# =============================================================================
# Transform Y and Split the data
# =============================================================================
# Transform Y label 
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

#Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, stratify=Y)

# =============================================================================
# Train the model
# =============================================================================

model = XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.001, 
                      use_label_encoder=False, eval_metric='logloss')
xgb_fit = model.fit(X_train, Y_train)

# =============================================================================
# Use K folds to improve sample representation
# =============================================================================
ten_fold = KFold(n_splits=10)
results = cross_val_score(model, X, Y)

print("K Fold Cross Validation results")
print("-"*90)
# Loop through results
for result in results:
    print("The resampled accuracy is: {:.2f}".format(result * 100))

mean_acc = results.mean()
print("The mean accuracy is {:.3f}".format(mean_acc * 100))

# =============================================================================
# Pickle model to work with it in next script
# =============================================================================
from pickle import dump, load
filename = "models/stranded_model.sav"
dump([model, X_train, X_test, Y_train, Y_test], open(filename, 'wb'))
