import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # Import train_test_split function_models
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# read the train and test dataset
titanic = pd.read_csv(r'train.csv')
df=pd.DataFrame(titanic)
print(df)
#show all columns
pd.set_option('display.max_columns', None)
print(titanic.head())
Features = ['Survived','Pclass', 'Sex', 'Age']
X=titanic[Features]
y=titanic.Survived
print(X)
X=X.drop('Survived',axis=1)
print(X)
X.replace({'male':0 , 'female':1},inplace=True)
print(X)
print(X.isnull().sum())
X.Age=X.Age.fillna(X.Age.mean())
print(X)
print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10, 20, 30, 100]
}
# Create Decision Tree classifer object
rf=RandomForestClassifier()
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Instantiate the grid search models
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0)
# Train Decision Tree Classifer
grid_search.fit(X_train,y_train)
predict_train = grid_search.predict(X_train)
print('\nTarget on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)
from pprint import pprint
best_grid = grid_search.best_estimator_
pprint(best_grid.get_params())

#from sklearn.svm import SVC
#param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
#grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
#grid.fit(X_train,y_train
