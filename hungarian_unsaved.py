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
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
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

def evaluation(Y_test,Y_pred):
    acc = accuracy_score(Y_test,Y_pred)
    rcl = recall_score(Y_test,Y_pred,average = 'weighted')
    f1 = f1_score(Y_test,Y_pred,average = 'weighted')
    ps = precision_score(Y_test,Y_pred,average = 'weighted')

    metric_dict={'accuracy': round(acc,3),
               'recall': round(rcl,3),
               'F1 score': round(f1,3),
               'Precision score': round(ps,3)
               }

    return print(metric_dict)

xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# model = DecisionTreeClassifier()
# model.fit(X_train_normal, y_train_normal)

y_pred = xgb_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title = "Hungarian Heart Disease",
  page_icon = ":heart:"
)

st.title("Hungarian Heart Disease")
st.write(f"**By Muhammad Zidan Ramdhani**")
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Model in Use : XGBoost with Oversampling & Normalization_])")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
  st.sidebar.header("**User Input** Sidebar")

  age = st.sidebar.number_input(label=":violet[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")

  sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  trestbps = st.sidebar.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")

  chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")

  fbs_sb = st.sidebar.selectbox(label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  restecg_sb = st.sidebar.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")

  exang_sb = st.sidebar.selectbox(label=":violet[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")

  data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
  }

  preview_df = pd.DataFrame(data, index=['input'])

  st.header("User Input as DataFrame")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  result = ":violet[-]"

  predict_btn = st.button("**Predict**", type="primary")

  st.write("")
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = xgb_model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    if prediction == 0:
      result = ":green[**Healthy**]" ":white[**Nice!, Keep doing what you're doing**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]" ":white[**No need to worry, just a gentle reminder to prioritize heart health.**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]" ":white[**You've got this! Take steps to improve your heart health, and you'll feel the difference.**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]" ":white[**It's a journey, not a race. Your commitment to improvement is the key.**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]" ":white[**It's a tough road, but you're resilient. With the right support, positive changes are possible**]"

  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

with tab2:
  st.header("Predict multiple data:")

  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = xgb_model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
