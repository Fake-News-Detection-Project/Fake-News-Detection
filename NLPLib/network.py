import torch
from torch import nn
from torch.nn import GRU, LSTM
import numpy as np


PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'

def load_glove_embeddings(embedding_file, padding_idx=0, padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD):
    """
    The function to load GloVe word embeddings
    
    :param      embedding_file:  The name of the txt file containing GloVe word embeddings
    :type       embedding_file:  str
    :param      padding_idx:     The index, where to insert padding and unknown words
    :type       padding_idx:     int
    :param      padding_word:    The symbol used as a padding word
    :type       padding_word:    str
    :param      unknown_word:    The symbol used for unknown words
    :type       unknown_word:    str
    
    :returns:   (a vocabulary size, vector dimensionality, embedding matrix, mapping from words to indices)
    :rtype:     a 4-tuple
    """
    word2index, embeddings, N = {}, [], 0
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0])
    
    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 2
        word2index[padding_word] = padding_idx
        word2index[unknown_word] = padding_idx + 1
                
    return N, D, np.array(embeddings, dtype=np.float32), word2index



class FakeNewsClassifier(nn.Module):
    def __init__(self, word_emb_file, hidden_size=100,
                 padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD, 
                 char_bidirectional=True, word_bidirectional=True, device='cpu'):
        
        super(FakeNewsClassifier, self).__init__()
        self.word_bidirectional = word_bidirectional
        self.hidden_size = hidden_size
        self.padding_word = padding_word
        self.unknown_word = unknown_word
        
        
        vocabulary_size, self.word_emb_size, embeddings, self.w2i = load_glove_embeddings(
            word_emb_file, padding_word=self.padding_word, unknown_word=self.unknown_word
        )
        
        self.word_emb = nn.Embedding(vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False)
        
        self.word_birnn = LSTM(
            self.word_emb_size,                             # input size
            self.hidden_size,                          # hidden size
            bidirectional=word_bidirectional,
            batch_first=True
        )
        
        multiplier = 2 if self.word_bidirectional else 1
        self.final_pred = nn.Linear(multiplier * self.hidden_size, 1)
        
        self.device = device
        self.to(device)
    
    def forward(self, x):
        """
        Performs a forward pass of a NER classifier
        Takes as input a 2D list `x` of dimensionality (B, T),
        where B is the batch size;
              T is the max sentence length in the batch (the sentences with a smaller length are already padded with a special token <PAD>)
        
        Returns logits, i.e. the output of the last linear layer before applying softmax.

        :param      x:    A batch of sentences
        :type       x:    list of strings
        """

                    
        #Get the word embedding
        wordIndices = torch.zeros( (len(x), len(x[0])) , dtype=torch.int, device=self.device)
        
        for batchIdx in range(len(x)): #PB ? all uppercase word are not in the w2i
            for t in range(len(x[0])):
                if x[batchIdx][t].lower() in self.w2i: wordIndices[batchIdx, t] = self.w2i[x[batchIdx][t].lower()]
                else: wordIndices[batchIdx, t] = self.w2i[self.unknown_word]
        
        wordEmb = self.word_emb(wordIndices)
        
        #Word level birnn
        (outputs, h_n) = self.word_birnn(wordEmb)
        h_n = h_n[0] #h_n is composed of h_n and c_n (we take just h_n)
                
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        return torch.sigmoid(self.final_pred(h_n))