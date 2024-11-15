import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
from utils import Net, Dataset, train_one_epoch, encode_labels
import warnings
warnings.filterwarnings("ignore")



x_train = pd.read_csv(r'/home/soham/Desktop/GitHub/Skybug-Movie-Genre-Classification/data/x_train.csv')
y_train = pd.read_csv(r'/home/soham/Desktop/GitHub/Skybug-Movie-Genre-Classification/data/y_train.csv')
y_train = y_train['Genre'].values

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
else:    
    device = torch.device('cpu')

print('Device set to - ',device)



#################################### MULTINOMIAL NAIVE BAYES ####################################

scaler = MinMaxScaler() 
scaler.fit(x_train)
x_t = scaler.transform(x_train)

NB = MultinomialNB(alpha=0.9)

print('Training Multinomial NB...')
NB.fit(x_t,y_train)

print("Training Accuracy: ",NB.score(x_t,y_train))
rep_nb = classification_report(y_train,NB.predict(x_t))
print(rep_nb)

with open('logs/classification_reports.txt', 'a') as f:
    f.write("Multinomial NB Classification Report:\n")
    f.write(rep_nb)
    f.write("\n\n")

pickle.dump(NB, open('models/NB.pkl','wb'))



#################################### SUPPORT VECTOR MACHINE (SVM) ####################################


# Sigmoid

svm_sig = SVC(C=1.0, kernel='sigmoid', degree=3, gamma='auto')

print('Training SVM Sigmoid...')
svm_sig.fit(x_train,y_train)

print("Training Accuracy: ",svm_sig.score(x_train,y_train))
rep_sig = classification_report(y_train,svm_sig.predict(x_train))
print(rep_sig)

with open('logs/classification_reports.txt', 'a') as f:
    f.write("SVM Sigmoid Classification Report:\n")
    f.write(rep_sig)
    f.write("\n\n")

pickle.dump(svm_sig, open('models/SVM_sig.pkl','wb'))



# RBF

svm_rbf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')

print('Training SVM RBF...')
svm_rbf.fit(x_train,y_train)

print("Training Accuracy: ",svm_rbf.score(x_train,y_train))
rep_rbf = classification_report(y_train,svm_rbf.predict(x_train))
print(rep_rbf)

with open('logs/classification_reports.txt', 'a') as f:
    f.write("SVM RBF Classification Report:\n")
    f.write(rep_rbf)
    f.write("\n\n")

pickle.dump(svm_rbf, open('models/SVM_rbf.pkl','wb'))



#################################### LOGISTIC REGRESSION ####################################

lr = LogisticRegression()

print('Training Logistic Regression...')
lr.fit(x_train,y_train)

print("Training Accuracy: ",lr.score(x_train,y_train))
rep_lr = classification_report(y_train,lr.predict(x_train))
print(rep_lr)

with open('logs/classification_reports.txt', 'a') as f:
    f.write("Logistic Regression Classification Report:\n")
    f.write(rep_lr)
    f.write("\n\n")

pickle.dump(lr, open('models/LR.pkl','wb'))



#################################### SGD REGRESSION ####################################

params = {
    'loss':['hinge','perceptron','log','modified_huber','squared_hinge','squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],
    'penalty':['l2','l1','elasticnet'],
    'alpha':[1e-3,1e-2,1e-1,1,10,100,1000],
    'max_iter':[5,10,15,20,25,30,35,40,45,50]
    }

sgd = SGDClassifier()
grid = GridSearchCV(estimator=sgd, param_grid=params, refit=True, cv=5, n_jobs=-1)

print('Training SGD Classifier...')
grid.fit(x_train,y_train)
print(grid.best_params_)

print("Training accuracy: ",grid.score(x_train,y_train))
rep_sgd = classification_report(y_train,grid.predict(x_train))
print(rep_sgd)

with open('logs/classification_reports.txt', 'a') as f:
    f.write("SGD Classification Report:\n")
    f.write(rep_sgd)
    f.write("\n\n")

pickle.dump(grid, open('models/SGD.pkl','wb'))




#################################### NEURAL NETWORK ####################################



y_train_coded = encode_labels(y_train)

train_dataset = Dataset(x_train.to_numpy().astype(np.float32), y_train_coded)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)


model = Net(384, 128, 64, 5, 1, device)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
batch_size = 1

all_losses = []

for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader, loss_fn, optimiser)
    print(f'Epoch {epoch} loss: {loss}')
    all_losses.append(loss)


pickle.dump(model, open('models/LSTM.pkl','wb'))