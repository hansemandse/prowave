import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Negative log-likelihood loss
def nll_loss(self, Y_hat, Y, vocab_size):
    
    # Remove the padding
    #Y = nn.utils.rnn.pack_padded_sequence(torch.squeeze(Y), lengths, batch_first = True, enforce_sorted = False) # This is set to make sure a not sorted sequence of lenghts is given.
    #Y, _ = nn.utils.rnn.pad_packed_sequence(Y, batch_first = True)
    
    # Flatten sequences within a batch
    Y = Y.view(-1)
    Y_hat = Y_hat.view(-1, vocab_size)

    # Filter out all padding tokens
    mask = (Y > 0).float()
    tokens = int(torch.sum(mask).data)
    Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

    # Return cross-entropy loss
    return -torch.sum(Y_hat) / tokens

# Perplexity
def plex_loss(self, Y_hat, Y, X_lengths):
    # Perplexity is just exponential cross-entropy
    return torch.exp(self.nll_loss(Y_hat, Y, X_lengths))

class ProLSTM(nn.Module):
    def __init__(self, lstm_layers = 1, lstm_hidden_size = 64, embedding_dim = 16, batch_size = 10, vocab_size = 30, clans = 10, families = 100): #1024
        super(ProLSTM, self).__init__()

        # Store values in this object
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Calculate total vocabulary size
        self.vocab_size = vocab_size
        self.total_size = vocab_size + clans + families

        # Build the network
        self.embed = nn.Embedding(
            num_embeddings = self.total_size,
            embedding_dim = self.embedding_dim,
            padding_idx = 0 # constant zero padding index
        )
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_layers,
            batch_first = True
        )
        self.ff = nn.Linear(
            in_features = self.lstm_hidden_size,
            out_features = self.vocab_size
        )

    def init_hidden(self):
        # Random initialization of hidden state
        hidden_a = torch.randn(self.lstm_layers, self.batch_size, self.lstm_hidden_size)
        hidden_b = torch.randn(self.lstm_layers, self.batch_size, self.lstm_hidden_size)

        # If the network is on the GPU, move the hidden state to the GPU as well
        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()
        
        # Wrap hidden state in variables
        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()

        # Calculate embedding
        X = self.embed(X)
        
        # Run through network
        X_LSTM = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first = True, enforce_sorted = False) # This is set to make sure a not sorted sequence of lenghts is given.
        X_LSTM, self.hidden = self.lstm(X_LSTM, self.hidden)
        X_LSTM, _ = nn.utils.rnn.pad_packed_sequence(X_LSTM, batch_first = True, padding_value = 0)
        
        # Make sure the padding is correct for the output - Batchsize - Sequence - LSTM Hidden
        # Batch Size, Max Seq in batch, LSTM Hidden - Padding such that it is Batch Size, Seqlen , LSTM Hidden
        # We need to add some Extra Padding Here
        X = X_LSTM
        
        X = X.view(batch_size, seq_len, self.lstm_hidden_size)
        
        # Run through linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.ff(X)

        # Softmax outputs
        X = F.log_softmax(X, dim = 1)
        X = X.view(batch_size, seq_len, self.vocab_size)
                            
        return X

class ProGRU(nn.Module):
    def __init__(self, gru_layers = 1, gru_hidden_size = 64, embedding_dim = 16, batch_size = 10, vocab_size = 30, clans = 10, families = 100): #1024
        super(ProGRU, self).__init__()

        # Store values in this object
        self.gru_layers = gru_layers
        self.gru_hidden_size = gru_hidden_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Calculate total vocabulary size
        self.vocab_size = vocab_size
        self.total_size = vocab_size + clans + families

        # Build the network
        self.embed = nn.Embedding(
            num_embeddings = self.total_size,
            embedding_dim = self.embedding_dim,
            padding_idx = 0 # constant zero padding index
        )
        self.gru = nn.GRU(
            input_size = self.embedding_dim,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_layers,
            batch_first = True
        )
        self.ff = nn.Linear(
            in_features = self.lstm_hidden_size,
            out_features = self.vocab_size
        )

    def init_hidden(self):
        # Random initialization of hidden state
        hidden_a = torch.randn(self.gru_layers, self.batch_size, self.gru_hidden_size)
        hidden_b = torch.randn(self.gru_layers, self.batch_size, self.gru_hidden_size)

        # If the network is on the GPU, move the hidden state to the GPU as well
        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()
        
        # Wrap hidden state in variables
        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()

        # Calculate embedding
        X = self.embed(X)
        
        # Run through network
        X_GRU = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first = True, enforce_sorted = False) # This is set to make sure a not sorted sequence of lenghts is given.
        X_GRU, self.hidden = self.gru(X_GRU, self.hidden)
        X_GRU, _ = nn.utils.rnn.pad_packed_sequence(X_GRU, batch_first = True, padding_value = 0)
        
        # Make sure the padding is correct for the output - Batchsize - Sequence - LSTM Hidden
        # Batch Size, Max Seq in batch, LSTM Hidden - Padding such that it is Batch Size, Seqlen , LSTM Hidden
        # We need to add some Extra Padding Here
        X = X_GRU
        
        X = X.view(batch_size, seq_len, self.gru_hidden_size)
        
        # Run through linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.ff(X)

        # Softmax outputs
        X = F.log_softmax(X, dim = 1)
        X = X.view(batch_size, seq_len, self.vocab_size)
                            
        return X
