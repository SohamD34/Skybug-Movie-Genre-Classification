import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sparse
import warnings
import matplotlib.pyplot as plt
from langdetect import detect
from googletrans import Translator
import re
import string
from sentence_transformers import SentenceTransformer
from utils import clean_text
warnings.filterwarnings('ignore')


''' ---------------------------- TRAINING DATA ------------------------------ '''



# So we have multiple genres of movies - drama, documentary, etc. 
# But out of those, the number of entries of a few genres is very high, which might be a cause of bias.

# We also have multiple languages in the description. 
# But the number is very small compared to English. Hence, for simplicity, we shall only consider those in English for now.

train_data = pd.read_csv('../data/train_data_with_languages.csv', index_col=0)
train_data = train_data[train_data['Language']=='en']
train_data['Summary'] = train_data['Summary'].apply(clean_text)
Y_train = train_data['Genre']

# The number of datapoints in "drama", "documentary","comedy", "short" is way larger and skewed compared to other classes. Hence we shall simply clump all the other classes together.

l = [' thriller ',' adult ',' crime ',' reality-tv ',' horror ',' sport ',' animation ',' action ',' fantasy ',' sci-fi ',' music ',' adventure ',' talk-show ',' western ',' family ',' mystery ',' history ',' news ',' biography ',' romance ',' game-show ',' musical ',' war ']
train_data["Genre"] = train_data["Genre"].replace(to_replace=l,value='other')
train_data["Genre"] = train_data["Genre"].replace(to_replace=[' drama '],value='drama')
train_data["Genre"] = train_data["Genre"].replace(to_replace=[' documentary '],value='documentary')
train_data["Genre"] = train_data["Genre"].replace(to_replace=[' comedy '],value='comedy')
train_data["Genre"] = train_data["Genre"].replace(to_replace=[' short '],value='short')

plt.figure(figsize=(6,5))
sns.countplot(data=Y_train,x=Y_train.values,palette='rocket')
sns.set(rc={'figure.figsize':(8,6)})
plt.xticks(rotation=45)
plt.show()


X_train = train_data["Summary"]
print("Before vectorisation: ",X_train.shape)

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(list(X_train))

print("After vectorisation:",sentence_embeddings.shape)

x_train = pd.DataFrame(sentence_embeddings)

x_train.to_csv('E:/SkyBug Technology Internship/Skybug-Movie-Genre-Classification/data/x_train.csv', index=False)
Y_train.to_csv('E:/SkyBug Technology Internship/Skybug-Movie-Genre-Classification/data/y_train.csv', index=False) 




''' ---------------------------- TESTING DATA ------------------------------ '''


test_data = pd.read_csv(r'E:/Skybug Technology Internship/Skybug-Movie-Genre-Classification/data/test_data_solution.txt', sep=":::", names=["Id","Title","Genre","Summary"])
X_test = test_data["Summary"]
Y_test = test_data["Genre"]

l = [' thriller ',' adult ',' crime ',' reality-tv ',' horror ',' sport ',' animation ',' action ',' fantasy ',' sci-fi ',' music ',' adventure ',' talk-show ',' western ',' family ',' mystery ',' history ',' news ',' biography ',' romance ',' game-show ',' musical ',' war ']
Y_test = Y_test.replace(to_replace=l,value='other')
Y_test = Y_test.replace(to_replace=[' drama '],value='drama')
Y_test = Y_test.replace(to_replace=[' documentary '],value='documentary')
Y_test = Y_test.replace(to_replace=[' comedy '],value='comedy')
Y_test = Y_test.replace(to_replace=[' short '],value='short')

print("Before vectorisation: ",X_test.shape)

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(list(X_test))

print("After vectorisation:",sentence_embeddings.shape)

X_test = pd.DataFrame(sentence_embeddings)
X_test.to_csv('E:/SkyBug Technology Internship/Skybug-Movie-Genre-Classification/data/x_test.csv', index=False)
Y_test.to_csv('E:/SkyBug Technology Internship/Skybug-Movie-Genre-Classification/data/y_test.csv', index=False) 