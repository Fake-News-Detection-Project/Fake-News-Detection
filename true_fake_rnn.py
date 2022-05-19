from NLPLib.DSP import RNNDataset, PadSequence
from NLPLib.network import FakeNewsClassifier
from tqdm import tqdm
import numpy as np
import argparse
from terminaltables import AsciiTable


from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import optim, nn
import torch




# Load the glove word embedder =========================================================================================


# The special symbols to be added at the end of strings
START_SYMBOL = '<START>'
END_SYMBOL = '<END>'

PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'






if __name__ == "__main__":
    
    #Args =============================================
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-ef', '--embeddings', default='', help='A file with word embeddings')
    parser.add_argument('-bs', '--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    
    
    is_cuda_available = torch.cuda.is_available()
    print("Is CUDA available? {}".format(is_cuda_available))
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Preparing datasets ==============================
    dataset = RNNDataset(lenSequence=100)
    training_loader = DataLoader(dataset, args.batch_size, collate_fn=PadSequence())
    
    
    # Prepare network training ========================
    network = FakeNewsClassifier("glove.6B.50d.txt", device=device, hidden_size=128)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
    
    network.train()
    
    for epoch in range( args.epochs ):
        lossAverage = 0
        for samples, labels in tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
            
            
            optimizer.zero_grad()
            
            logits = network(samples)
            logits_shape = logits.shape
                                  
            loss = criterion(logits[:,0], torch.tensor(labels, dtype=torch.float).to(device))
            loss.backward()
            
            lossAverage += loss.item()
        
            clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            
        print(f"Epoch average loss: {lossAverage / len(training_loader)}")
    
    # Evaluation
    network.eval()
    confusion_matrix = [[0, 0],
                        [0, 0]]

    dataset.setTraning = False
    testing_loader = DataLoader(dataset, args.batch_size, collate_fn=PadSequence())
    
    for x, y in tqdm(testing_loader, desc="Testing set"):
        result = network(x)
        pred = torch.round(result).cpu().detach().numpy().reshape(-1,)
        y = np.array(y)

        tp = np.sum(pred[y == 1])
        tn = np.sum(1 - pred[y == 0])
        fp = np.sum(1 - y[pred == 1])
        fn = np.sum(y[pred == 0])

        confusion_matrix[0][0] += tn
        confusion_matrix[1][1] += tp
        confusion_matrix[0][1] += fp
        confusion_matrix[1][0] += fn

    table = [['', 'Predicted Fake', 'Predicted True'],
             ['Real Fake', confusion_matrix[0][0], confusion_matrix[0][1]],
             ['Real True', confusion_matrix[1][0], confusion_matrix[1][1]]]

    t = AsciiTable(table)
    print(t.table)
    print("Accuracy: {}".format(
        round((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix), 4))
    )
