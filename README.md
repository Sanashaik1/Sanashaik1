Submission Title: Heart Disease Data Analysis and Visualization

Code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loading and Preprocessing
df = pd.read_csv('heart_disease.csv')
df.fillna(df.mean(), inplace=True)
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Data Analysis
avg_age_with_hd = df[df['target'] == 1]['age'].mean()
avg_age_without_hd = df[df['target'] == 0]['age'].mean()
cp_dist = df['cp'].value_counts()
corr_coef = df['thalach'].corr(df['age'])
sex_hd_corr = df['sex'].corr(df['target'])

# Data Visualization
plt.hist(df['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.bar(cp_dist.index, cp_dist.values)
plt.title('Chest Pain Distribution')
plt.xlabel('Chest Pain Type')
plt.ylabel('Frequency')

plt.scatter(df['thalach'], df['age'])
plt.title('Thalach vs Age')
plt.xlabel('Thalach')
plt.ylabel('Age')

plt.boxplot([df[df['target'] == 1]['age'], df[df['target'] == 0]['age']])
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Heart Disease Status')
plt.ylabel('Age')

# Advanced Analysis (using numpy)
corr_matrix = np.corrcoef(df.select_dtypes(include=[np.number]).values.T)
chol_rolling_mean = df['chol'].rolling(window=5).mean()

# Bonus
def predict_hd(patient_data):
    if patient_data['age'] > 60 and patient_data['thalach'] > 150 and patient_data['chol'] > 200:
        return 1
    else:
        return 0

fig, axs = plt.subplots(2, 2)
axs[0][0].hist(df['age'], bins=20)
axs[0][1].bar(cp_dist.index, cp_dist.values)
axs[1][0].scatter(df['thalach'], df['age'])
axs[1][1].boxplot([df[df['target'] == 1]['age'], df[df['target'] == 0]['age']])
plt.show()
