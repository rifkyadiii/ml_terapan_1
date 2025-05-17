#!/usr/bin/env python
# coding: utf-8

# Import Library

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report, 
                            roc_curve, auc)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')


# Set random seed

# In[42]:


np.random.seed(42)


# 1. Data Loading

# In[43]:


try:
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    print("Data loaded successfully.")
except:
    print("Warning: Could not load data from local file.")
    print("Simulating the dataset for demonstration purposes.")

    np.random.seed(42)
    n = 5110
    df = pd.DataFrame({
        'id': np.arange(1, n+1),
        'gender': np.random.choice(['Male', 'Female'], size=n, p=[0.41, 0.59]),
        'age': np.random.normal(loc=43, scale=22, size=n),
        'hypertension': np.random.binomial(1, 0.1, size=n),
        'heart_disease': np.random.binomial(1, 0.05, size=n),
        'ever_married': np.random.choice(['Yes', 'No'], size=n),
        'work_type': np.random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], size=n),
        'Residence_type': np.random.choice(['Urban', 'Rural'], size=n),
        'avg_glucose_level': np.random.normal(loc=106, scale=45, size=n),
        'bmi': np.random.normal(loc=28.5, scale=7.5, size=n),
        'smoking_status': np.random.choice(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], size=n),
        'stroke': np.random.binomial(1, 0.05, size=n)
    })


# In[44]:


df['age'] = np.clip(df['age'], 0.1, 100)
df['bmi'] = np.clip(df['bmi'], 10, 60)


# In[45]:


high_risk = ((df['age'] > 60) | (df['hypertension'] == 1) | (df['avg_glucose_level'] > 200))
df.loc[high_risk, 'stroke'] = np.random.binomial(1, 0.15, size=sum(high_risk))


# In[46]:


random_indices = np.random.choice(df.index, size=int(len(df)*0.04), replace=False)
df.loc[random_indices, 'bmi'] = np.nan


# 2. Exploratory Data Analysis

# In[47]:


df.describe()


# In[48]:


df.info()


# In[49]:


print("\nBasic Information:")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1]-1}") 


# In[50]:


print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# In[51]:


print("\nTarget variable distribution:")
print(df['stroke'].value_counts())
print(f"Stroke percentage: {df['stroke'].mean()*100:.2f}%")


# In[52]:


plt.figure(figsize=(10, 6))
sns.countplot(x='stroke', data=df)
plt.title('Distribution of Stroke Cases')
plt.show()


# In[53]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='stroke', bins=30, element='step')
plt.title('Age Distribution by Stroke Status')
plt.show()


# In[54]:


plt.figure(figsize=(15, 10))
for i, feature in enumerate(['age', 'avg_glucose_level', 'bmi']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='stroke', y=feature, data=df)
    plt.title(f'{feature} by Stroke Status')
plt.tight_layout()
plt.show()


# In[55]:


categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                   'work_type', 'Residence_type', 'smoking_status']

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    stroke_rate = df.groupby(col)['stroke'].mean() * 100
    stroke_rate.plot(kind='bar')
    plt.title(f'Stroke Rate (%) by {col}')
    plt.ylabel('Stroke Rate (%)')
    plt.show()


# In[56]:


# Correlation matrix
plt.figure(figsize=(12, 10))
df_corr = df.copy()
for col in categorical_cols:
    df_corr[col] = pd.factorize(df_corr[col])[0]


# In[57]:


corr_matrix = df_corr.drop('id', axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# 3. Data Preparation

# In[58]:


# Drop ID column
df = df.drop('id', axis=1)


# In[59]:


df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 120], labels=['0-20', '20-40', '40-60', '60-80', '80+'])


# In[60]:


# Handle missing values
df['bmi'] = df.groupby(['gender', 'age_group'])['bmi'].transform(lambda x: x.fillna(x.median()))
df['bmi'] = df['bmi'].fillna(df['bmi'].median())


# In[61]:


df = df.drop('age_group', axis=1)


# In[62]:


# Handle outliers
bmi_99_percentile = df['bmi'].quantile(0.99)
df['bmi'] = np.where(df['bmi'] > bmi_99_percentile, bmi_99_percentile, df['bmi'])


# In[63]:


# Feature encoding
# One-hot encoding untuk categorical features
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# In[64]:


# Feature scaling
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])


# In[65]:


# Feature engineering
df_encoded['age_bmi'] = df_encoded['age'] * df_encoded['bmi']
df_encoded['age_glucose'] = df_encoded['age'] * df_encoded['avg_glucose_level']
df_encoded['hypertension_heart'] = df_encoded['hypertension'] * df_encoded['heart_disease']


# 4. Train-Test Split

# In[66]:


X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[67]:


# Check class distribution
print(f"Training set stroke distribution: {np.bincount(y_train)}")
print(f"Testing set stroke distribution: {np.bincount(y_test)}")


# 5. Handle imbalanced data menggunakan SMOTE

# In[68]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Training set stroke distribution: {np.bincount(y_train_resampled)}")


# In[69]:


# Feature selection
selector = SelectFromModel(XGBClassifier(random_state=42))
selector.fit(X_train_resampled, y_train_resampled)
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {', '.join(selected_features)}")


# In[70]:


X_train_selected = selector.transform(X_train_resampled)
X_test_selected = selector.transform(X_test)


# 6. Modeling

# In[71]:


# Logistic Regression
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]


# In[72]:


# Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]


# In[73]:


# XGBoost
xgb = XGBClassifier(scale_pos_weight=len(y_train_resampled) / sum(y_train_resampled), random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]


# 7. Hyperparameter tuning for XGBoost

# In[74]:


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(scale_pos_weight=len(y_train_resampled) / sum(y_train_resampled), random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")


# In[75]:


best_xgb = grid_search.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)
y_proba_best_xgb = best_xgb.predict_proba(X_test)[:, 1]


# 8. Model Evaluation

# In[76]:


def evaluate_model(y_true, y_pred, y_proba, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC-ROC: {roc_auc:.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# In[77]:


# Evaluate all models
lr_metrics = evaluate_model(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")
rf_metrics = evaluate_model(y_test, y_pred_rf, y_proba_rf, "Random Forest")
xgb_metrics = evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost")
best_xgb_metrics = evaluate_model(y_test, y_pred_best_xgb, y_proba_best_xgb, "XGBoost (tuned)")


# In[78]:


# Compare models
models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'XGBoost (tuned)']
metrics = [lr_metrics, rf_metrics, xgb_metrics, best_xgb_metrics]


# In[79]:


metrics_df = pd.DataFrame(metrics, index=models)
print("\nModel Comparison:")
print(metrics_df)


# 9. ROC Curve Visualization

# In[80]:


plt.figure(figsize=(10, 8))

# Plot ROC curve for each model
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
fpr_best_xgb, tpr_best_xgb, _ = roc_curve(y_test, y_proba_best_xgb)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_metrics["roc_auc"]:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_metrics["roc_auc"]:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_metrics["roc_auc"]:.3f})')
plt.plot(fpr_best_xgb, tpr_best_xgb, label=f'XGBoost tuned (AUC = {best_xgb_metrics["roc_auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()


# 10. Feature Importance

# In[81]:


plt.figure(figsize=(12, 8))
feature_importance = rf.feature_importances_

sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

