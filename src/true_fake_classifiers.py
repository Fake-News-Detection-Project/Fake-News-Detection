from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score



## Dataloading
data_repo = "../data"
corpus_filename = "cleaned_corpus.csv"

corpus = pd.read_csv(os.path.join(data_repo, corpus_filename))
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['final_text'], corpus['label'],test_size=0.2)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000, use_idf=False)
Tfidf_vect.fit(corpus['final_text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

SVM = svm.SVC()
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_NB = SVM.predict(Test_X_Tfidf)

print("SVM Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
