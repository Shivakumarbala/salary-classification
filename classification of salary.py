#import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib .pyplot as plt
%matplotlib inline

# Importing dataset
dataset = pd.read_excel('adult.xlsx')

#preview dataset
dataset.head()
dataset.shape
dataset.info()
dataset.describe().T
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
round((dataset.isin(['?']).sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
income = dataset['salary'].value_counts(normalize=True)
round(income * 100, 2).astype('str') + ' %'
dataset = dataset.replace('?', np.nan)
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
columns_with_nan = ['workclass', 'occupation', 'native-country']
for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace = True)
from sklearn.preprocessing import LabelEncoder
for col in dataset.columns:
  if dataset[col].dtypes == 'object':         
    encoder = LabelEncoder()         
    dataset[col] = encoder.fit_transform(dataset[col])
X = dataset.drop('salary', axis = 1) 
Y = dataset['salary']
#to avoid the problem of overfitting
from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(random_state = 42)
selector.fit(X, Y)
feature_imp = selector.feature_importances_
for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))
X = X.drop(['workclass', 'education', 'race', 'sex', 'capital-loss', 'native-country'], axis = 1)
from sklearn.preprocessing import StandardScaler
for col in X.columns:     
  scaler = StandardScaler()     
  X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
from sklearn.preprocessing import StandardScaler
for col in X.columns:     
  scaler = StandardScaler()     
  X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
round(Y.value_counts(normalize=True) * 100, 2).astype('str') + ' %'
#balancing a dataset
from imblearn.over_sampling import RandomOverSampler 
ros = RandomOverSampler(random_state = 42)
ros.fit(X, Y)
X_resampled, Y_resampled = ros.fit_resample(X, Y)
round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.2, random_state = 42)
print("X_train shape:", X_train.shape) 
print("X_test shape:", X_test.shape) 
print("Y_train shape:", Y_train.shape) 
print("Y_test shape:", Y_test.shape)
from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state = 42)
ran_for.fit(X_train, Y_train)
Y_pred_ran_for = ran_for.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print('Random Forest Classifier:')
print('Accuracy score:',round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:',round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))
#constructing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred_ran_for,labels=[0,1])
print(cm)
sns.heatmap(cm,cmap="Greens",annot=True)
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()
