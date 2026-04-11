# 0. import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. read data
df = pd.read_csv("data_churn.csv", sep = ",")

# CLEANING DATA (bỏ customerID)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce') 
df['TotalCharges'] = df['TotalCharges'].fillna(0)

bin_cols = ["gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling"]
for col in bin_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
bin_cols += ["SeniorCitizen"]

df['Churn_num'] = df['Churn'].map({'No': 0, 'Yes': 1})
# print("SeniorCitizen vs Churn:")
# print(df.groupby('SeniorCitizen')['Churn_num'].mean())

# print("PhoneService vs Churn:")
# print(df.groupby('PhoneService')['Churn_num'].mean())
bin_cols.remove('PhoneService')  

cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod']

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# 2. split
X = df.drop(columns = ['Churn', 'customerID', 'PhoneService'])
Y = df['Churn'].map({'No': 0, 'Yes': 1})
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# 3. preprocessing
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

X_train_bin = X_train[bin_cols].values
X_test_bin = X_test[bin_cols].values

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# 4. linking
X_train_final = np.hstack([X_train_num, X_train_cat, X_train_bin])
X_test_final = np.hstack([X_test_num, X_test_cat, X_test_bin])

# 5. train
model = LogisticRegression(max_iter = 2000, class_weight = 'balanced', random_state = 42)
model.fit(X_train_final, Y_train)

# 6. Predict
Y_pred = model.predict(X_test_final)

# 7. Metrics
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print("F1 Score:", f1_score(Y_test, Y_pred))

# DIAGRAM
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ['No Churn', 'Churn']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# [[TN FP]
#  [FN TP]]