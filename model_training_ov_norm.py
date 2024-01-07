import pandas as pd
import numpy as np
import re
import itertools
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix,precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import time

with open("data/hungarian.data", encoding='Latin') as file:
    lines = [line.strip() for line in file]

data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

df = df.iloc[:,:-1]
df = df.drop(df.columns[0], axis=1)

df = df.astype(float)

df.replace(-9.0, np.nan, inplace=True)

df.isnull().sum()

df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39,42,49,56]]

column_mapping = {
    2: 'age',
    3: 'sex',
    8: "cp",
    9: "trestbps",
    11: "chol",
    15: "fbs",
    18: "restecg",
    31: 'thalach',
    37: 'exang',
    39: 'oldpeak',
    40: 'slope',
    43: "ca",
    50: 'thal',
    57: 'target'

}

df_selected.rename(columns=column_mapping, inplace=True)

df_selected.value_counts()

df_selected.isnull().sum()

columns_to_drop = ['ca', 'slope', 'thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

df_selected.isnull().sum()

meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

fill_values = {'trestbps': meanTBPS, 'chol': meanChol, 'fbs': meanfbs, 'thalach': meanthalach, 'exang': meanexang, 'restecg': meanRestCG}
dfClean = df_selected.fillna(value=fill_values)

# X = dfClean.drop("target",axis=1).values
# y = dfClean.iloc[:,-1]

X = dfClean.drop("target", axis=1)
y = dfClean['target']

# oversampling
smote = SMOTE(random_state=42)
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

plt.figure(figsize=(12,4))

new_df1 = pd.DataFrame(data=y)

plt.subplot(1,2,2)
new_df2 = pd.DataFrame(data=y_smote_resampled)

new_df1 = pd.DataFrame(data=y)
new_df1.value_counts()

scaler = MinMaxScaler()

X_smote_resampled_normal = scaler.fit_transform(X_smote_resampled)

dfcek1 = pd.DataFrame(X_smote_resampled_normal)

#membagi fitur dan target menjadi data train dan test (untuk yang oversampled saja)
X_train, X_test, y_train, y_test = train_test_split(X_smote_resampled, y_smote_resampled, test_size=0.2, random_state=42,stratify=y_smote_resampled)

# membagi fitur dan target menjadi data train dan test (untuk yang oversample + normalization)
X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(X_smote_resampled_normal, y_smote_resampled, test_size=0.2, random_state=42,stratify = y_smote_resampled)

# Mendefinisikan fungsi evaluasi dengan parameter Y_test (kelas sebenarnya) dan Y_pred (kelas prediksi)
def evaluation(Y_test, Y_pred):
    # Menghitung nilai akurasi
    acc = accuracy_score(Y_test, Y_pred)

    # Menghitung nilai recall dengan menggunakan rata-rata tertimbang
    rcl = recall_score(Y_test, Y_pred, average='weighted')

    # Menghitung nilai F1 score dengan menggunakan rata-rata tertimbang
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    # Menghitung nilai precision dengan menggunakan rata-rata tertimbang
    ps = precision_score(Y_test, Y_pred, average='weighted')

    # Membuat dictionary yang berisi hasil evaluasi
    metric_dict = {
        'accuracy': round(acc, 3),
        'recall': round(rcl, 3),
        'F1 score': round(f1, 3),
        'Precision score': round(ps, 3)
    }

    # Menampilkan hasil evaluasi
    return print(metric_dict)

# ============================================== KNN Oversampling + Normalization ==============================================

# Membuat model K-Nearest Neighbors (KNN) dengan jumlah tetangga (n_neighbors) sebanyak 3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Melatih model KNN menggunakan data latih yang telah dinormalisasi (X_train_normal dan y_train_normal)
knn_model.fit(X_train_normal, y_train_normal)

# Menggunakan model KNN yang telah dilatih untuk membuat prediksi pada data uji yang telah dinormalisasi (X_test_normal)
y_pred_knn = knn_model.predict(X_test_normal)

# Evaluasi model KNN yang telah dinormalisasi
print("K-Nearest Neighbors (KNN) Model:")
accuracy_knn_smote_normal = round(accuracy_score(y_test_normal, y_pred_knn), 3)
print("Accuracy:", accuracy_knn_smote_normal)

# Mencetak laporan klasifikasi untuk model KNN yang telah dinormalisasi
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_knn))

# Menggunakan fungsi evaluasi untuk mengevaluasi model KNN yang telah dioptimalkan pada data uji yang telah dinormalisasi
evaluation(y_test_normal, y_pred_knn)

# ============================================== KNN Oversampling + Normalization ==============================================


# ============================================== Random Forest Oversampling + Normalization ==============================================

# Membuat model Random Forest dengan 100 pohon keputusan (n_estimators=100)
# dan menggunakan nilai seed acak (random_state=42) untuk hasil yang dapat direproduksi
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Melatih model Random Forest menggunakan data latih yang telah dinormalisasi (X_train_normal dan y_train_normal)
rf_model.fit(X_train_normal, y_train_normal)

# Menggunakan model Random Forest yang telah dilatih untuk membuat prediksi pada data uji yang telah dinormalisasi (X_test_normal)
y_pred_rf = rf_model.predict(X_test_normal)

# Evaluasi model Random Forest yang telah dinormalisasi
print("\nRandom Forest Model:")

# Menghitung dan mencetak akurasi model Random Forest
accuracy_rf_smote_normal = round(accuracy_score(y_test_normal, y_pred_rf), 3)
print("Accuracy:", accuracy_rf_smote_normal)

# Mencetak laporan klasifikasi untuk model Random Forest yang telah dinormalisasi
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_rf))

# Menggunakan fungsi evaluasi untuk mengevaluasi model Random Forest yang telah dioptimalkan pada data uji yang telah dinormalisasi
evaluation(y_test_normal,y_pred_rf)

# ============================================== Random Forest Oversampling + Normalization ==============================================


# ============================================== XGBoost Oversampling + Normalization ==============================================

# Membuat model XGBoost dengan learning rate 0.1, 100 pohon keputusan (n_estimators=100),
# dan menggunakan nilai seed acak (random_state=42) untuk hasil yang dapat direproduksi
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)

# Melatih model XGBoost menggunakan data latih yang telah dinormalisasi (X_train_normal dan y_train_normal)
xgb_model.fit(X_train_normal, y_train_normal)

# Menggunakan model XGBoost yang telah dilatih untuk membuat prediksi pada data uji yang telah dinormalisasi (X_test_normal)
y_pred_xgb = xgb_model.predict(X_test_normal)

# Evaluasi model XGBoost yang telah dinormalisasi
print("\nXGBoost Model:")

# Menghitung dan mencetak akurasi model XGBoost
accuracy_xgb_smote_normal = round(accuracy_score(y_test_normal, y_pred_xgb), 3)
print("Accuracy:", accuracy_xgb_smote_normal)

# Mencetak laporan klasifikasi untuk model XGBoost yang telah dinormalisasi
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_xgb))

# Menggunakan fungsi evaluasi untuk mengevaluasi model XGBoost yang telah dioptimalkan pada data uji yang telah dinormalisasi
evaluation(y_test_normal, y_pred_xgb)

# ============================================== XGBoost Oversampling + Normalization ==============================================

# Menyimpan model KNN ke dalam file menggunakan pickle
with open("knn_norm_ov_model.pkl", 'wb') as file:
    pickle.dump(knn_model, file)

# Menyimpan model Random Forest ke dalam file menggunakan pickle  
with open("rf_norm_ov_model.pkl", 'wb') as file:
    pickle.dump(rf_model, file)

# Menyimpan model XGBoost ke dalam file menggunakan pickle 
with open("xgb_norm_ov_model.pkl", 'wb') as file:
    pickle.dump(xgb_model, file)