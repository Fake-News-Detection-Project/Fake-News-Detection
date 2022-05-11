import torch
import pandas as pd
import os
import nltk
import numpy as np
import re

from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    data_repo = "../data"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    # Adding labels
    true = pd.read_csv(os.path.join(data_repo, true_filename))
    true['label'] = np.ones(len(true), dtype=int)
    fake = pd.read_csv(os.path.join(data_repo, fake_filename))        
    fake['label'] = np.zeros(len(fake), dtype=int)

    data = pd.concat((true,fake),axis=0)
    
    lemmatizer = nltk.stem.WordNetLemmatizer()

    stop_words = ["a", "about", "an", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "of", "on", "or", "that", "the", "these", "this", "too", "was", "what", "when", "where", "who", "will"]
    new_text = []
    pattern = "[^a-zA-Z]"
    for txt in data.text:
        txt = re.sub(pattern," ",txt)
        txt = txt.lower()
        txt = nltk.word_tokenize(txt) # Tokenizing
        txt = [lemmatizer.lemmatize(word) for word in txt]
        txt = " ".join(txt)
        new_text.append(txt)

    print(new_text[0])

    
    








