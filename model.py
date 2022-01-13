import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
import pickle
import plotly.express as px

df = pd.read_csv('C:\\Users\\user\\Downloads\\heart.csv')
features = df.iloc[:,:-1]
labels = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.8,stratify = labels)
def scores(x_train,x_test,y_train,y_test):
    acc_metrics = pd.DataFrame(columns = ['Models','Accuracies'])
    acc_metrics.Models = ['Logistic Regression','Decision Tree Classifier','Random Forest Classifier','ADA Boost Classifier','Gradient Boost Classifier','Support Vector Classifier']
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    logreg_pred = logreg.predict(x_test)
    dt = DecisionTreeClassifier()
    dt.fit(x_train,y_train)
    dt_pred = dt.predict(x_test)
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    rf_pred = rf.predict(x_test)
    ada = AdaBoostClassifier()
    ada.fit(x_train,y_train)
    ada_pred = ada.predict(x_test)
    gb = GradientBoostingClassifier()
    gb.fit(x_train,y_train)
    gb_pred = gb.predict(x_test)
    sv = SVC()
    sv.fit(x_train,y_train)
    sv_pred = sv.predict(x_test)
    acc_metrics.Accuracies = [metrics.accuracy_score(y_test,logreg_pred),metrics.accuracy_score(y_test,dt_pred),metrics.accuracy_score(y_test,rf_pred),metrics.accuracy_score(y_test,ada_pred),metrics.accuracy_score(y_test,gb_pred),metrics.accuracy_score(y_test,sv_pred)]
    return acc_metrics
# accuracy_scores = scores(x_train,x_test,y_train,y_test)
# accuracy_scores  # Random Forest works best so lets hypertune it

param_grid = [
{'n_estimators': [10, 25], 'max_features': [5, 10], 
 'max_depth': [10, 50, None], 'bootstrap': [True, False]}
]

#rfc = RandomForestClassifier()

#gridsv = GridSearchCV(rfc,param_grid,scoring='accuracy')
#gridsv.fit(x_train,y_train)
#final_model = RandomForestClassifier(max_depth = 10,max_features = 5,n_estimators = 10,bootstrap = False)
#final_model.fit(x_train,y_train)
#final_model.predict(x_test)
#pickle.dump(final_model,open('HeartDiseasePredictor','wb'))