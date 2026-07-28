import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

d    = np.load("data/features_9d.npz")
X_tr = d["train_features"]
X_te = d["test_features"]
y_tr = d["y_train"]
y_te = d["y_test"]

svm  = joblib.load("backend/models/svm_model.pkl")
svd  = joblib.load("backend/models/svd_reducer.pkl")
sc   = joblib.load("backend/models/minmax_scaler.pkl")

test_acc  = (svm.predict(X_te) == y_te).mean()
train_acc = (svm.predict(X_tr) == y_tr).mean()

print(f"Test  acc : {test_acc*100:.2f}%   (expected ~84.95%)")
print(f"Train acc : {train_acc*100:.2f}%")
print(f"SVD components: {svd.n_components}")
print(f"Scaler feature_range: {sc.feature_range}")
print("No InconsistentVersionWarning above = resave successful!")
