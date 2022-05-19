import os
import string
from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from nltk.corpus import stopwords

from tqdm import tqdm
import numpy as np
import json
import nltk

  

#Interface
class DatasetInterface:
    def getSample(self, training:bool) -> Tuple[str, int]:
        """Load a random sample from the dataset."""
        pass

    def getLength(self, training:bool) -> int:
        """Get the length of the dataset."""
        pass


class Dataset1(DatasetInterface):
    def __init__(self, true_filepath = "NLPLib/Datasets/Dataset1/True.csv", fake_filepath="NLPLib/Datasets/Dataset1/Fake.csv", train_size = 0.8, seed:int = None):
        true = pd.read_csv(true_filepath)
        true['label'] = np.ones(len(true), dtype=int)
        fake = pd.read_csv(fake_filepath)
        fake['label'] = np.zeros(len(fake), dtype=int)

        self.corpus = pd.concat((true,fake),axis=0)
        
        self.seed = np.random.randint(0, 10000) if seed == None else seed
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.corpus.text, self.corpus.label, train_size=train_size, random_state=seed)
        

    def getSample(self, training: bool = True, testing=False, returnLabel:bool = True) -> Tuple[np.ndarray, float]:

        if training:
            for i in range(self.getLength(True)):
                yield (self.x_train.iloc[i], self.y_train.iloc[i]) if returnLabel else self.x_train.iloc[i]

        if testing:
            for i in range(self.getLength(False)):
                yield (self.x_test.iloc[i], self.y_test.iloc[i]) if returnLabel else self.x_test.iloc[i]
    
    def getLabel(self, training:bool = True):
        for i in range(self.getLength(training)):
            if training:
                yield self.y_train.iloc[i]
            else:
                yield self.y_test.iloc[i]
        
                
    def getLength(self, training = True):
        return len(self.x_train) if training else len(self.x_test)
    


class CleanedDataset1(DatasetInterface):
    def __init__(self, file_path = "NLPLib/Datasets/CleanedDataset1/cleaned_corpus.csv", train_size = 0.8, seed:int = None):

        self.corpus = pd.read_csv(file_path)
        
        self.seed = np.random.randint(0, 10000) if seed == None else seed
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.corpus.final_text, self.corpus.label, train_size=train_size, random_state=seed)
        


    def getSample(self, training: bool = True, testing=False, returnLabel:bool = True) -> Tuple[np.ndarray, float]:

        if training:
            for i in range(self.getLength(True)):
                yield (self.x_train.iloc[i], self.y_train.iloc[i]) if returnLabel else self.x_train.iloc[i]

        if testing:
            for i in range(self.getLength(False)):
                yield (self.x_test.iloc[i], self.y_test.iloc[i]) if returnLabel else self.x_test.iloc[i]
        
    
    
    
    def getLabel(self, training:bool = True):
        for i in range(self.getLength(training)):
            if training:
                yield self.y_train.iloc[i]
            else:
                yield self.y_test.iloc[i]
        
                
    def getLength(self, training = True):
        return len(self.x_train) if training else len(self.x_test)
    
PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'

class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """
    def __call__(self, batch, pad_data=PADDING_WORD, max_len=500, pad_labels=0):
        batch_data, batch_labels = zip(*batch)

        padded_data = [[b[i] if i < len(b) else pad_data for i in range(max_len)] for b in batch_data]
        return padded_data, batch_labels
    
    
    

class RNNDataset(Dataset):
    def __init__(self, true_filepath = "NLPLib/Datasets/Dataset1/True.csv", fake_filepath="NLPLib/Datasets/Dataset1/Fake.csv", lenSequence=500, corpus_percent = 1, setTraining = True, train_size = 0.8, preprocess=True):

        self.setTraning = setTraining
        
        true = pd.read_csv(true_filepath)
        true['label'] = np.ones(len(true), dtype=int)
        fake = pd.read_csv(fake_filepath)
        fake['label'] = np.zeros(len(fake), dtype=int)

        self.corpus = pd.concat((true,fake),axis=0)
        self.lenSequence = lenSequence
        self.corpus_percent = corpus_percent
        self.preprocess = preprocess
        
        self.stop_words = set(stopwords.words('english'))
        # self.stop_words |= set(["reuters", "washington", "seattle", "lima", "vatican", "york", "zurich"])
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        
        x, y = self.corpus.text.to_numpy(), self.corpus.label.to_numpy()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=train_size)
        
                

    def __getitem__(self, idx) -> Tuple[np.ndarray, float]:        
        if self.setTraning:
            txt = self.x_train[idx]

            sequence = self.applyPreprocess(txt)
            
            return sequence, self.y_train[idx]
        else:
            txt = self.x_test[idx]
            sequence = self.applyPreprocess(txt)
            return sequence, self.y_test[idx]
                
    def __len__(self):
        if self.setTraning:
            return (int)(len(self.y_train) * self.corpus_percent)
        else:
            return (int)(len(self.y_test))
        
    def applyPreprocess(self, txt):
        subStringIndex = txt.find(" - ") #To get rid of the reuters and name of the city
        if self.preprocess and subStringIndex != -1 and subStringIndex < 100: txt = txt[subStringIndex:] 
        
        if self.preprocess: txt = txt.translate(self.punctuation_translator) #We get rid of the punctuation
        sequence = nltk.word_tokenize(txt)
        if self.preprocess: sequence = [w.lower() for w in sequence if not w.lower() in self.stop_words] #Removal of stop words
        
        return sequence




#Testing the DSP (execute in NLPLib folder)
if __name__ =="__main__":
    data_repo = "../data"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    dataset = Dataset1(true_filepath=os.path.join(data_repo, true_filename), fake_filepath=os.path.join(data_repo, fake_filename))
    nbTrue = 0
    nbFalse = 0
    for (text, label) in tqdm(dataset.getSample(), desc="Reading corpus", total=dataset.getLength()):
        if label == 0: nbFalse += 1
        if label == 1: nbTrue += 1
    
    print(f"{nbTrue=}")
    print(f"{nbFalse=}")

