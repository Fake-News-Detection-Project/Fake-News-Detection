import torch
from torch import nn
from torch.nn import GRU, LSTM
import numpy as np


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh, device='cpu'):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation
        self.hidden_size = hidden_size

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        
        
        self.device = device
        self.to(device)

        
    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell


        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """

        xTransformed = (self.w_ih @ x.T).T + self.b_ih
        hTransformed = (self.w_hh @ h.T).T + self.b_hh
        
        sigSum = torch.sigmoid(xTransformed + hTransformed)
        
        n = self.activation(xTransformed[:, self.hidden_size * 2:] + sigSum[:, :self.hidden_size] * hTransformed[:, self.hidden_size * 2:])
                
        return (1 - sigSum[:, self.hidden_size:self.hidden_size*2]) * n + sigSum[:, self.hidden_size:self.hidden_size*2] * h


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False, device='cpu'):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation, device=device) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation, device=device) # backward cell
        self.hidden_size = hidden_size
        
        self.device = device
        self.to(device)
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """

        outputs = torch.zeros((x.shape[0], x.shape[1], 2 * self.hidden_size if self.bidirectional else self.hidden_size))
        h_fw = torch.zeros((x.shape[0], x.shape[2]))
        h_bw = torch.zeros((x.shape[0], x.shape[2]))
        
        
        #Forward pass
        h = torch.zeros((x.shape[0], self.hidden_size), device=self.device)
        for t in range(x.shape[1]):
            h = self.fw.forward(x[:, t], h)
            outputs[:, t, :self.hidden_size] = h
        
        h_fw = h
        
        #Backward pass
        if self.bidirectional:
            h = torch.zeros((x.shape[0], self.hidden_size), device=self.device)
            for t in range(x.shape[1] - 1, -1, -1):
                h = self.bw.forward(x[:, t], h)
                outputs[:, t, self.hidden_size:] = h
            
            h_bw = h
                        
            return (outputs, h_fw, h_bw)
        
        return (outputs, h_fw)


def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"

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
        self.final_pred = nn.Linear(multiplier * self.hidden_size, 2)
        
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
        h_n = h_n[0]
        h_n = torch.reshape(h_n, (h_n.shape[1], h_n.shape[0] * h_n.shape[2]) )
        return self.final_pred(h_n)