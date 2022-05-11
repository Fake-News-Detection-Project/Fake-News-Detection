from email.policy import default
import torch
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk import ngrams

from tqdm import tqdm
import numpy as np
import re
from collections import defaultdict
# nlkt.download('stopwords')

class DataPrep:
    def __init__(self, true_filepath, fake_filepath, n=2, method='TF', feature_size=1000):
        self.n = n
        self.method = method
        self.feature_size = feature_size

        true = pd.read_csv(true_filepath)
        true['label'] = np.ones(len(true), dtype=int)
        fake = pd.read_csv(fake_filepath)        
        fake['label'] = np.zeros(len(fake), dtype=int)

        self.corpus = pd.concat((true,fake),axis=0)[:] # DEBUG remove [:100] for prod

        self.stemmer = porter.PorterStemmer()

        self.i2g = [] # index to gram : [ gram0, gram1, gram2, ]
        self.g2i = dict() # gram to index : { gram0:0, gram1:1, gram2:2, }
        self.gram_count = [] # count # of grams apparition indexwise [ # gram0, # gram1, # gram2, ]
        self.N = 0 # total number of != grams over all corpus
        self.total_grams = 0 # number of grams 

    def preprocess(self):
         ## Data preprocessing
        clean_text, n_grams_text = [], []
        pattern = "[^a-zA-Z]" # Pattern for only keeping letters 
        stop_words = set( stopwords.words('english') )
    
        for txt in tqdm(self.corpus.text, desc="Processing corpus - Cleaning data"): 
            txt = re.sub(pattern," ",txt) # Remove punctuation / non letter characters
            txt = txt.lower() # case lowering
            txt = nltk.word_tokenize(txt) # Tokenizing
            txt = [ self.stemmer.stem(word) for word in txt if not word in stop_words ] # Stop words removal + stemming
            txt = ' '.join(txt)
            txt = list(pad_both_ends(txt.split(), n=self.n)) # padding for computing n-grams
            txt = list(ngrams(txt, n=self.n)) # computing n-grams

            for gram in txt: # Counting grams for all texts
                self.total_grams += 1
                if self.g2i.get(gram) == None: 
                    self.i2g.append( gram )
                    self.gram_count.append(0)
                    self.g2i[gram] = self.N  * 1000
                    self.N  += 1
                self.gram_count[ self.g2i[gram] ] += 1

            clean_text.append(txt)

        # Adding a column to the dataset
        self.corpus["grams"] = clean_text

        print( self.gram_count[np.argmax(self.gram_count)]  )
        selected_grams_idx = np.flip( np.argsort( self.gram_count ) )[:self.feature_size] # Extracting the most frequent grams
        print(selected_grams_idx)

        selected_grams = [ self.i2g[i] for i in selected_grams_idx ] # List of the most import 
        selected_grams_dict = dict( zip(selected_grams, [i for i in range(self.feature_size)]) )
        
        for gram in selected_grams:
            print(gram, self.gram_count[ self.g2i[gram] ])
        if self.method == 'TF':
            feature_vect = np.zeros( ( len(self.corpus), self.feature_size) )
            
            for i, gram_list in enumerate( tqdm(self.corpus.grams, desc="Processing corpus - TF Feature extraction\n") ):
                for gram in gram_list:
                    if selected_grams_dict.get(gram) != None:
                        
                        feature_vect[i, selected_grams_dict.get(gram)] += 1
                # print(feature_vect[i])
                if np.sum(feature_vect[i])  != 0:
                    feature_vect[i] /= np.sum(feature_vect[i])
                else:
                    print("IS NAN")


                

if __name__ == '__main__':
    ## Dataloading
    data_repo = "../data"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    n = 1
    dataObject = DataPrep( true_filepath=os.path.join(data_repo, true_filename), fake_filepath=os.path.join(data_repo, fake_filename), n=n, method='TF', feature_size= 50)
    dataObject.preprocess()
   
    








