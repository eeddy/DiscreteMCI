import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random
import pickle

"""
This is a basic Discrete classifier that goes from EMG to prediction. It uses cross entropy loss to 
optimize its performance on predicting the correct active class.
"""
class DiscreteClassifier(nn.Module):
    def __init__(self, emg_size, cnn=True, file_name=None, temporal_hidden_size=128, temporal_layers=3, mlp_layers=[128, 64, 32], n_classes=6, type='GRU', conv_kernel_sizes = [3, 3, 3], conv_out_channels=[16, 32, 64]):
        super().__init__()
        
        fix_random_seed(0)
        self.cnn = cnn
        self.file_name = file_name
        self.log = {
            'tr_loss': [],
            'te_loss': [],
            'tr_acc': [],
            'te_acc': [],
            'tr_acc_a': [],
            'te_acc_a': [],
        }
        self.min_loss = 100000

        dropout = 0.2 

        if self.cnn:
            self.conv_layers = nn.ModuleList()
            in_channels = emg_size[1]  # Channels in EMG signal
            for i in range(len(conv_out_channels)):
                self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels[i], kernel_size=conv_kernel_sizes[i], padding='same'))
                self.conv_layers.append(nn.BatchNorm1d(conv_out_channels[i]))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
                self.conv_layers.append(nn.Dropout(dropout))
                in_channels = conv_out_channels[i] 

        spoof_emg_input = torch.zeros((1, *emg_size))
        if self.cnn:
            conv_out = self.forward_conv(spoof_emg_input)
            conv_out_size = conv_out.shape[-1] 
        else:
            conv_out_size = emg_size[1]
            conv_out = spoof_emg_input
        
        # Set the temporal feature extraction piece
        if type == 'LSTM':
            self.temporal = nn.LSTM(conv_out_size, temporal_hidden_size, num_layers=temporal_layers, batch_first=True, dropout=dropout)
        elif type == 'BILSTM':
            self.temporal = nn.LSTM(conv_out_size, temporal_hidden_size, num_layers=temporal_layers, batch_first=True, dropout=dropout, bidirectional=True)
        elif type == 'RNN':
            self.temporal = nn.RNN(conv_out_size, temporal_hidden_size, num_layers=temporal_layers, batch_first=True, dropout=dropout, nonlinearity='relu')
        elif type == 'GRU':
            self.temporal = nn.GRU(conv_out_size, temporal_hidden_size, num_layers=temporal_layers, batch_first=True, dropout=dropout)
        elif type == 'TRANSFORMER':
            self.pos_encoder = PositionalEncoding(conv_out_size)
            self.temporal = nn.Sequential(
                nn.LayerNorm(conv_out_size),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=conv_out_size, nhead=8, dim_feedforward=temporal_hidden_size, dropout=dropout, batch_first=True
                    ),
                    num_layers=3
                ),
                nn.LayerNorm(conv_out_size)
            )
        else:
            print("Invalid selection of model type.")
            exit(1)

        emg_output_shape = self.forward_temporal(conv_out).numel()

        self.initial_layer = nn.Linear(emg_output_shape, mlp_layers[0])
        self.layer1 = nn.Linear(mlp_layers[0], mlp_layers[1])
        self.layer2 = nn.Linear(mlp_layers[1], mlp_layers[2])
        self.output_layer = nn.Linear(mlp_layers[-1], n_classes) 
        self.relu = nn.ReLU()

    def forward_conv(self, x):
        batch_size, seq_len, channels, samples = x.shape

        # Reshape for 1D Conv: (batch*seq, channels, samples)
        x = x.view(batch_size * seq_len, channels, samples)

        for layer in self.conv_layers:
            x = layer(x)

        # Reshape back for temporal layer: (batch, seq, features)
        _, channels_out, samples_out = x.shape
        x = x.view(batch_size, seq_len, channels_out * samples_out)

        return x

    def forward_mlp(self, x):
        out = self.initial_layer(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out 


    def forward_once(self, emg, emg_len=None):
        if self.cnn:
            out = self.forward_conv(emg)
        else:
            out = emg 
        out = self.forward_temporal(out, emg_len)
        out = self.forward_mlp(out)
        return out  

    def forward_temporal(self, emg, lengths=None):
        # Resetting LSTM Memory - I don't find it really matters online because randomization from batches 
        # batch_size = emg.shape[0]
        # h0 = torch.zeros(3, batch_size, 256).to(emg.device)
        # c0 = torch.zeros(3, batch_size, 256).to(emg.device)
        # out, _ = self.temporal(emg, (h0, c0))

        if isinstance(self.temporal, nn.Sequential):
            emg = self.pos_encoder(emg)
            out = self.temporal(emg)
        else:
            out, _ = self.temporal(emg)

        if lengths is not None:
            out = torch.stack([s[lengths[i]-1] for i,s in enumerate(out)])
        else:
            out = out[:,-1,:]
        return out 
    
    def _run_epoch(self, device, dl, optimizer, loss_function, train=False):
        loss_arr = []
        acc_arr = []
        acc_a_arr = [] 
        for data, labels, lengths in dl:
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
           
            output = self.forward_once(data, lengths)

            loss = loss_function(output, labels) 
            loss_arr.append(loss.item())

            acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
            acc_arr.append(acc.item())
            acc_a = sum((torch.argmax(output, 1) == labels) & (labels != 0)) / sum(labels != 0)
            acc_a_arr.append(acc_a.item())
            
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1) # Needs to be on for Transformer 
                optimizer.step()

        return loss_arr, acc_arr, acc_a_arr

    def fit(self, tr_dl, te_dl, learning_rate=1e-4, epochs=20):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate,)
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.9 ** (epoch - warmup_epochs)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loss_function = nn.CrossEntropyLoss()

        # now start the training
        for epoch in range(0, epochs):
            #training set
            self.train()
            tr_loss, tr_acc, tr_acc_a = self._run_epoch(device, tr_dl, optimizer, loss_function, train=True)
            self.eval()
            te_loss, te_acc, te_acc_a = self._run_epoch(device, te_dl, optimizer, loss_function, train=False)

            if np.mean(te_loss) < self.min_loss:
                print("Improved validation loss... Saving model.")
                torch.save(self, 'Results/' + self.file_name + '.model')
                self.min_loss = np.mean(te_loss)

            scheduler.step()

            # Log everything 
            self.log['tr_loss'].append(np.mean(tr_loss))
            self.log['te_loss'].append(np.mean(te_loss))
            self.log['tr_acc'].append(np.mean(tr_acc))
            self.log['te_acc'].append(np.mean(te_acc))
            self.log['tr_acc_a'].append(np.mean(tr_acc_a))
            self.log['te_acc_a'].append(np.mean(te_acc_a))

            print(f"{epoch}: trloss:{np.mean(tr_loss):.4f} tracc:{np.mean(tr_acc):.4f} tracc_a:{np.mean(tr_acc_a):.4f} teloss:{np.mean(te_loss):.4f} teacc:{np.mean(te_acc):.4f} teacc_a:{np.mean(te_acc_a):.4f}")

        self.eval()

        if self.file_name is not None:
            pickle.dump(self.log, open('Results/' + self.file_name + '.pkl', 'wb'))
    
    def predict(self, x, device='cpu'):
        self.to(device)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to(device))
        return np.array([p.argmax().item() for p in preds])


def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model)) # Learnable parameters

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x