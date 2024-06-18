import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error , matthews_corrcoef, recall_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, average_precision_score

embbedings = pd.read_csv("peptides_embs.csv")
rg_data = pd.read_csv("peptides_rg.csv")

rg_data=rg_data[["peptide_id","mean_Average_Rg"]]

merged_df= pd.merge(rg_data,embbedings,left_on="peptide_id",right_on="peptide_id",how="inner")
merged_df.to_csv("MLdata.csv")


# Custom function for RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a custom RMSE scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)



X = merged_df.drop(["peptide_id","mean_Average_Rg"],axis=1)
X=X.astype(float)
y = merged_df["mean_Average_Rg"]
y = y.astype(float)
model = linear_model.LinearRegression()
print("LinearRegression")
print(cross_val_score(model, X, y, cv=3,scoring=rmse_scorer))


model = linear_model.RidgeCV()
print("RidgeCV")
print(cross_val_score(model, X, y, cv=3,scoring=rmse_scorer))

model = linear_model.LassoCV()
print("LassoCV")
print(cross_val_score(model, X, y, cv=3),scoring=rmse_scorer)
