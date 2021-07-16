import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'healthcare-dataset-stroke-data.csv')
print(df.columns)
print(df.info())
#df.hist(figsize = (20, 20))
#plt.show()
print(df.isnull().sum())
df['bmi'].fillna(df['bmi'].mean(),inplace=True)
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

gender=enc.fit_transform(df['gender'])
smoking_status=enc.fit_transform(df['smoking_status'])
work_type=enc.fit_transform(df['work_type'])
Residence_type=enc.fit_transform(df['Residence_type'])
ever_married=enc.fit_transform(df['ever_married'])
df['ever_married']=ever_married
df['Residence_type']=Residence_type
df['smoking_status']=smoking_status
df['gender']=gender
df['work_type']=work_type
print(df[['ever_married', 'Residence_type', 'smoking_status', 'gender', 'work_type']].head())
print (df.info())

X = df.drop('stroke', axis=1)
y = df['stroke']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
