import math
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Negative log-likelihood loss from https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
def nll_loss(Y_hat, Y, vocab_size):
    # Flatten sequences within a batch
    Y = Y.view(-1)
    Y_hat = Y_hat.view(-1, vocab_size)

    # Filter out all padding tokens
    mask = (Y > 0).float()
    tokens = int(torch.sum(mask).item())
    Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

    # Return cross-entropy loss
    return -torch.sum(Y_hat) / tokens

# Perplexity
def plex_loss(Y_hat, Y, X_lengths):
    # Perplexity is just exponential cross-entropy
    return torch.exp(nll_loss(Y_hat, Y, X_lengths))

# Training loop
def train(net, train_loader, valid_loader, use_pretrained = None, keep_training = False, vocab_size = 30, criterion = nll_loss, epochs = 500):
    """
    Trains a net with the given data or fetches a pretrained model

    Args:
     `net`: the model to train
     `train_loader`: a DataLoader object with training data
     `valid_loader`: a DataLoader object with validation data
     `use_pretrained`: one of None or a path to a pretrained model
     `keep_training`: if True, the pretrained model will be further trained, otherwise the pretrained model will be returned
     `vocab_size`: size of the output vocabulary, default 30
     `criterion`: the loss function, default nll_loss
     `epochs`: the number of epochs to train for, default 500

    Returns a trained model
    """
    if use_pretrained is not None:
        if type(use_pretrained) == str:
            net.load_state_dict(torch.load(use_pretrained))
            if not keep_training:
                return net
        else:
            raise ValueError("Expected use_pretrained to be one of None or str")

    optimizer = optim.Adam(net.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)

    # For tracking intermediate values
    training_loss = []
    validation_loss = []
    
    # If CUDA is available, move net to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # Training loop - first set the network into training mode
    for i in range(epochs):
        # Validate network
        validation_loss.append(evaluate(net, valid_loader))
        
        # Train network
        net.train()
        epoch_training_loss = 0
        
        # Start Training
        for inputs, targets in train_loader:
            # Forward pass
            optimizer.zero_grad()
            inputs.to(device)
            if type(net) == ProLSTM or type(net) == ProGRU:
                # To calculate forward pass, we must calculate the original sequence lengths of the
                # input tensors without padding characters
                input_lengths = [sum(k > 0) for k in inputs] 
                output = net(inputs, torch.tensor(input_lengths))
            elif type(net) == ProTrans:
                # To calculate forward pass, we must calculate a mask for the source input
                src_mask = net.generate_square_subsequent_mask(inputs.size(1))
                output = net(inputs, src_mask)
                targets = targets.T.contiguous()
            output.to(device)
            batch_loss = criterion(output, targets, vocab_size)

            # Back-propagation and weight update
            batch_loss.backward()
            optimizer.step() 

            # Update loss
            epoch_training_loss += batch_loss.item()

        # Step the scheduler
        scheduler.step()
        
        # Save loss for plot
        training_loss.append(epoch_training_loss / len(train_loader))

        # Print loss every epoch
        print(f'Epoch {i}, training loss: {training_loss[-1]}, Validation Perplexity Loss: {(validation_loss[-1])}')
        
        # Print the Learning Rate Being Used
        print('Learning Rate: {}'.format(scheduler.get_lr()))

    # Plot training and validation loss
    epoch = np.arange(len(training_loss))
    plt.figure(figsize=(10,5))
    plt.plot(epoch, training_loss, 'r', label='Training loss',)
    plt.plot(epoch, validation_loss, 'b', label='Validation Perplexity loss',)
    plt.legend()
    plt.xlabel('Epoch'), plt.ylabel('NLL')
    plt.show()

    return net.cpu()

# Evaluation
def evaluate(net, loader, criterion = plex_loss, vocab_size = 30):
    """
    Evaluates a net on the given test data

    Args:
     `net`: the model to evaluate
     `loader`: a Dataloader object
    
    Returns total loss on the test data
    """
    # First, set the network into evaluation mode
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if type(net) == ProLSTM or type(net) == ProGRU:
                input_lengths = [sum(k > 0) for k in inputs] 
                output = net(inputs, torch.tensor(input_lengths))
            elif type(net) == ProTrans:
                src_mask = net.generate_square_subsequent_mask(inputs.size(1))
                output = net(inputs, src_mask)
            else: # For the WaveNet do this:
                input_lengths = [sum(k > 0) for k in inputs] 
                output = net(inputs)
            batch_loss = criterion(output, targets, vocab_size)
            total_loss += batch_loss.item()
    return total_loss / len(loader)

class ProLSTM(nn.Module):
    def __init__(self, lstm_layers = 1, lstm_hidden_size = 256, embedding_dim = 32, batch_size = 50, vocab_size = 30, clans = 10, families = 100): #1024
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
        self.DropOut = nn.Dropout(p=0.2)

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
        X_LSTM, _ = nn.utils.rnn.pad_packed_sequence(X_LSTM, batch_first = True, padding_value = 0, total_length = 514)
        
        # Make sure the padding is correct for the output - Batchsize - Sequence - LSTM Hidden
        # Batch Size, Max Seq in batch, LSTM Hidden - Padding such that it is Batch Size, Seqlen , LSTM Hidden
        # We need to add some Extra Padding Here
        X = X_LSTM
        _, seq_len, _ = X.size()
        
        # Add Dropout  - 20 % 
        X = self.DropOut(X)
        
        #X = X.view(batch_size, seq_len, self.lstm_hidden_size)
        
        # Run through linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.ff(X)

        # Softmax outputs
        X = F.log_softmax(X, dim = 1)
        X = X.view(batch_size, seq_len, self.vocab_size)
                            
        return X

class ProGRU(nn.Module):
    def __init__(self, gru_layers = 1, gru_hidden_size = 288, embedding_dim = 32, batch_size = 50, vocab_size = 30, clans = 10, families = 100): #1024
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
            hidden_size = self.gru_hidden_size,
            num_layers = self.gru_layers,
            batch_first = True
        )
        self.ff = nn.Linear(
            in_features = self.gru_hidden_size,
            out_features = self.vocab_size
        )
        
        self.DropOut = nn.Dropout(p=0.2)

    def init_hidden(self):
        # Random initialization of hidden state
        hidden = torch.randn(self.gru_layers, self.batch_size, self.gru_hidden_size)

        # If the network is on the GPU, move the hidden state to the GPU as well
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        
        # Wrap hidden state in variables
        hidden = Variable(hidden)

        return hidden

    def forward(self, X, X_lengths):
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()

        # Calculate embedding
        X = self.embed(X)
        
        # Run through network
        X_GRU = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first = True, enforce_sorted = False) # This is set to make sure a not sorted sequence of lenghts is given.
        X_GRU, self.hidden = self.gru(X_GRU, self.hidden)
        X_GRU, _ = nn.utils.rnn.pad_packed_sequence(X_GRU, batch_first = True, padding_value = 0, total_length = 514)
        
        # Make sure the padding is correct for the output - Batchsize - Sequence - LSTM Hidden
        # Batch Size, Max Seq in batch, LSTM Hidden - Padding such that it is Batch Size, Seqlen , LSTM Hidden
        # We need to add some Extra Padding Here
        X = X_GRU
        _, seq_len, _ = X.size()
        
        #X = X.view(batch_size, seq_len, self.gru_hidden_size)
        
        # Add Dropout  - 20 % 
        X = self.DropOut(X)
        
        # Run through linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.ff(X)

        # Softmax outputs
        X = F.log_softmax(X, dim = 1)
        X = X.view(batch_size, seq_len, self.vocab_size)
                            
        return X

# The following is modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class ProTrans(nn.Module):
    def __init__(self, nintoken, nouttoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(ProTrans, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(nintoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nouttoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src.T) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# The following is modified from https://github.com/Dankrushen/Wavenet-PyTorch
class ProWaveNet(nn.Module):
    def __init__(self, 
                 num_time_samples,
                 num_channels = 1,
                 num_classes  = 30,
                 num_blocks   = 2,
                 num_layers   = 14,
                 num_hidden   = 128,
                 kernel_size  = 2,
                 vocab_size = 30,
                 clans = 10,
                 families = 100):
        super(ProWaveNet, self).__init__()
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.receptive_field = 1 + (kernel_size - 1) * \
                               num_blocks * sum([2**k for k in range(num_layers)])
        self.output_width = num_time_samples - self.receptive_field + 1
        #print('receptive_field: {}'.format(self.receptive_field))
        #print('Output width: {}'.format(self.output_width))
        
        self.set_device()

        hs = []
        batch_norms = []

        # add gated convs
        first = True
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                if first:
                    h = GatedResidualBlock(num_channels, num_hidden, kernel_size, 
                                           self.output_width, dilation=rate)
                    first = False
                else:
                    h = GatedResidualBlock(num_hidden, num_hidden, kernel_size,
                                           self.output_width, dilation=rate)
                h.name = 'b{}-l{}'.format(b, i)

                hs.append(h)
                batch_norms.append(nn.BatchNorm1d(num_hidden))

        self.hs = nn.ModuleList(hs)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.relu_1 = nn.ReLU()
        self.conv_1_1 = nn.Conv1d(num_hidden, num_hidden, 1)
        self.relu_2 = nn.ReLU()
        self.conv_1_2 = nn.Conv1d(num_hidden, num_hidden, 1)
        # For the number of classes we might just do an FFNN instead - this way we get the dimension we need
        self.h_class = nn.Conv1d(num_hidden, num_classes, 1) # Had Kernel Size 2 before        
        # Embedding
        self.embedding_dim = 30
        
        # Calculate total vocabulary size
        self.vocab_size = vocab_size
        self.total_size = vocab_size + clans + families

        # Build the network
        self.embed = nn.Embedding(
            num_embeddings = self.total_size,
            embedding_dim = self.embedding_dim,
            padding_idx = 0 # constant zero padding index
        )
        
        

    def forward(self, x):
        skips = []
        x = self.embed(x)
        # Permute the imput to have the dimension [Batch x Channel x Len] - here the channels are the same as number of tokens/embeddings
        x = x.permute(0, 2, 1)
        
        #print("Shape of X:", x.shape )
        
        for layer, batch_norm in zip(self.hs, self.batch_norms):
            x, skip = layer(x)
            x = batch_norm(x)
            skips.append(skip)

        x = reduce((lambda a, b : torch.add(a, b)), skips)
        x = self.relu_1(self.conv_1_1(x))
        x = self.relu_2(self.conv_1_2(x))

        x = self.h_class(x)

        # Take Softmax on the output - the signal is permuted therefore dimension 1
        return F.log_softmax(x, dim = 1)

    def set_device(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train_WaveNet(self, dataloader, valid_loader, num_epochs=25, validation=False, disp_interval=None, vocab_size = 30):
        self.to(self.device)

        if validation:
            phase = 'Validation'
        else:
            phase = 'Training'

        losses = []
        validation_loss = []
        for epoch in range(1, num_epochs + 1):
            
            # Validate network
            validation_loss.append(evaluate(self, valid_loader))
            
            if not validation:
                super().train()
            else:
                self.eval()
                
            # reset loss for current phase and epoch
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # track history only during training phase
                with torch.set_grad_enabled(not validation):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels, vocab_size)
                    
                    if not validation:
                        loss.backward()
                        self.optimizer.step()
                        
                running_loss += loss.item()


            # Step The Scheduler
            self.scheduler.step()
            
            losses.append(running_loss / len(dataloader))
            
            if disp_interval is not None and epoch % disp_interval == 0:
                epoch_loss = running_loss / len(dataloader)
                print(f'Epoch {epoch}, training loss: {losses[-1]}, Validation Perplexity Loss: {(validation_loss[-1])}')
                print('Learning Rate: {}'.format(self.scheduler.get_lr()))
  
        epochtotal = np.arange(len(losses))
        plt.figure(figsize=(10,5))
        plt.plot(epochtotal, losses, 'r', label='Training loss',)
        plt.plot(epochtotal, validation_loss, 'b', label='Validation Perplexity Loss',)
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('NLL')
        plt.show()    

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        padding = self.dilation - (x.shape[-1] + self.dilation - 1) % self.dilation
        x = nn.functional.pad(x, (self.dilation, 0))
        return torch.mul(self.conv_f(x), self.sig(self.conv_g(x)))

class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_width, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedResidualBlock, self).__init__()
        self.output_width = output_width
        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        skip = self.conv_1(self.gatedconv(x))
        residual = torch.add(skip, x)

        skip_cut = skip.shape[-1] - self.output_width
        skip = skip.narrow(-1, skip_cut, self.output_width)
        return residual, skip

class Generator(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def _shift_insert(self, x, y):
        x = x.narrow(-1, y.shape[-1], x.shape[-1] - y.shape[-1])
        dims = [1] * len(x.shape)
        dims[-1] = y.shape[-1]
        y = y.reshape(dims)
        return torch.cat([x, self.dataset._to_tensor(y)], -1)

    def tensor2numpy(self, x):
        return x.data.numpy()

    def predict(self, x):
        x = x.to(self.model.device)
        self.model.to(self.model.device)
        return self.model(x)

    def run(self, x, num_samples, disp_interval=None):
        x = self.dataset._to_tensor(self.dataset.preprocess(x))
        x = torch.unsqueeze(x, 0)

        y_len = self.dataset.y_len
        out = np.zeros((num_samples // y_len + 1) * y_len)
        n_predicted = 0
        for i in range(num_samples // y_len + 1):
            if disp_interval is not None and i % disp_interval == 0:
                print('Sample {} / {}'.format(i * y_len, num_samples))

            y_i = self.tensor2numpy(self.predict(x).cpu())
            y_i = self.dataset.label2value(y_i.argmax(axis=1))[0]
            y_decoded = self.dataset.encoder.decode(y_i)

            out[n_predicted:n_predicted + len(y_decoded)] = y_decoded
            n_predicted += len(y_decoded)

            # shift sequence and insert generated value
            x = self._shift_insert(x, y_i)

        return out[0:num_samples]
