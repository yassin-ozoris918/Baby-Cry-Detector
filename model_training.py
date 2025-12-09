import os, joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from configs.config import MODEL_OUT_DIR
from data_preparation.dataset_builder import dataset_builder


print("=== Step 1: Build dataset ===")
X, y = dataset_builder()


# Encode labels    ==> cry = 1, not_cry = 0
print("\n=== Step 2: Encode labels ===")
le = LabelEncoder()
y_enc = le.fit_transform(y)



# Balance Dataset
print("\n=== Step 3: Balance dataset with SMOTE ===")
sm = SMOTE(random_state=42)

X_bal, y_bal = sm.fit_resample(X, y_enc)



# Normalization
print("\n=== Step 4: Normalize features ===")
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_bal)


# Train_Test Split
print("\n=== Step 5: Split into train/test sets ===")
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_bal, test_size=0.2, random_state=42)



# Train SVM
print("\n=== Step 6: Train SVM classifier ===")
svm = SVC(kernel='rbf', C=5, gamma='scale')
svm.fit(X_train, y_train)



# Evaluate
print("\n=== Step 7: Evaluate model ===")
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Save
print("\n=== Step 8: Save model, scaler, and label encoder ===")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)
joblib.dump(svm, f"{MODEL_OUT_DIR}/svm_cry_detector.pkl")
joblib.dump(scaler, f"{MODEL_OUT_DIR}/feature_scaler.pkl")
joblib.dump(le, f"{MODEL_OUT_DIR}/label_encoder.pkl")


print("Model saved!")

