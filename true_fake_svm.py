from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import numpy as np
from nltk.stem import porter
import nltk
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    ## Dataloading
    data_repo = "NLPLib/Datasets/Dataset1"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    true = pd.read_csv(os.path.join(data_repo, true_filename) )
    true['label'] = np.ones(len(true), dtype=int)
    fake = pd.read_csv(os.path.join(data_repo, fake_filename))        
    fake['label'] = np.zeros(len(fake), dtype=int)

    corpus = pd.concat((true,fake),axis=0)[:]

    pattern = "[^a-zA-Z]"
    stop_words = set( stopwords.words('english') )
    n = 1
    stemmer = porter.PorterStemmer()
    lengths = []
    new_txt = []
    for index,txt in tqdm(enumerate(corpus['text']), desc="Preprocessing data", total=len(corpus)):
        txt = re.sub(pattern," ", txt) # Remove punctuation / non letter characters
        txt = txt.lower() # case lowering
        txt = nltk.word_tokenize(txt) # Tokenizing
        txt = [ stemmer.stem(word) for word in txt if not word in stop_words ] # Stop words removal + stemming
        lengths.append(len(txt))

        new_txt.append(str(txt))
    corpus["final_text"] = new_txt

    print("Lengths\n", "Mean", np.mean(lengths), "Var", np.std(lengths))
    
    corpus.to_csv(r'NLPLib/Datasets/CleanedDataset1/cleaned_corpus.csv', index=False)

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['final_text'], corpus['label'],test_size=0.2)

    Tfidf_vect = TfidfVectorizer(max_features=5000, use_idf=False)
    Tfidf_vect.fit(corpus['final_text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    SVM = svm.SVC()
    SVM.fit(Train_X_Tfidf,Train_Y)
    
    predictions_NB = SVM.predict(Test_X_Tfidf)

    print("SVM Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
