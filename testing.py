import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier


###################### MULTINOMIAL NAIVE BAYES ######################

model = pickle.load(open('models/NB.pkl','rb'))
x_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

y_pred = model.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))


###################### SUPPORT VECTOR MACHINES ######################

# RBF

svm_rbf = pickle.load(open('models/SVM_rbf.pkl','rb'))
x_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

y_pred = svm_rbf.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))


# SIGMOID

svm_sig = pickle.load(open('models/SVM_sig.pkl','rb'))
x_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

y_pred = svm_sig.predict(x_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))


###################### LINEAR REGRESSION ######################

x_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')
lr = pickle.load(open('models/LR.pkl','rb'))

y_pred = lr.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))