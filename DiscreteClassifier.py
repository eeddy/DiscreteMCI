import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random
import pickle
from torch.utils.data import DataLoader, Dataset

"""
This is a basic Discrete classifier that goes from EMG to prediction. It uses cross entropy loss to 
optimize its performance on predicting the correct active class.
"""
class DiscreteClassifier(nn.Module):
    def __init__(self, emg_size, cnn=True, file_name=None, temporal_hidden_size=128, temporal_layers=3, mlp_layers=[128, 64, 32], n_classes=5, type='GRU', conv_kernel_sizes = [3, 3, 3], conv_out_channels=[16, 32, 64]):
        super().__init__()
        
        fix_random_seed(0)
        self.cnn = cnn
        self.file_name = file_name
        self.threshold = 0.5
        self.best_metric = 0
        self.log = {
            'tr_loss': [],
            'te_loss': [],
            'tr_precision': [],
            'te_precision': [],
            'tr_recall': [],
            'te_recall': [],
            'tr_micro_f1': [],
            'te_micro_f1': [],
        }
        self.min_loss = 0

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
            self.input_projection = nn.Linear(conv_out_size, temporal_hidden_size)
            self.pos_encoder = PositionalEncoding(temporal_hidden_size)
            self.temporal = nn.Sequential(
                nn.LayerNorm(temporal_hidden_size),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=temporal_hidden_size, nhead=8, dim_feedforward=temporal_hidden_size*4, dropout=dropout, batch_first=True
                    ),
                    num_layers=temporal_layers
                ),
                nn.LayerNorm(temporal_hidden_size)
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
        if isinstance(self.temporal, nn.Sequential):
            emg = self.input_projection(emg)
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
        precisions, recalls, f1s = [], [], []

        for data, labels, lengths in dl:
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)  # float32 multi-hot, shape [B, C]
            lengths = lengths.to(device)

            logits = self.forward_once(data, lengths)        # [B, C], raw logits
            loss = loss_function(logits, labels)             # BCEWithLogitsLoss
            loss_arr.append(loss.item())

            # Metrics: thresholded predictions
            preds = (torch.sigmoid(logits) >= self.threshold).float()

            # Micro precision/recall/F1 over all labels
            tp = (preds * labels).sum().item()
            fp = (preds * (1 - labels)).sum().item()
            fn = ((1 - preds) * labels).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)  # keep for Transformer
                optimizer.step()

        return loss_arr, precisions, recalls, f1s

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

        loss_function = nn.BCEWithLogitsLoss()

        # now start the training
        for epoch in range(0, epochs):
            # train
            self.train()
            tr_loss, tr_prec, tr_rec, tr_f1 = self._run_epoch(device, tr_dl, optimizer, loss_function, train=True)

            # eval
            self.eval()
            te_loss, te_prec, te_rec, te_f1 = self._run_epoch(device, te_dl, optimizer, loss_function, train=False)

            # model selection on micro-F1
            mean_te_f1 = np.mean(te_f1)
            if mean_te_f1 > self.best_metric:
                print("Improved testing F1... Saving model.")
                if self.file_name is not None:
                    torch.save(self, self.file_name + '.model')
                self.best_metric = mean_te_f1

            scheduler.step()

            # Log everything
            self.log['tr_loss'].append(np.mean(tr_loss))
            self.log['te_loss'].append(np.mean(te_loss))
            self.log['tr_precision'].append(np.mean(tr_prec))
            self.log['te_precision'].append(np.mean(te_prec))
            self.log['tr_recall'].append(np.mean(tr_rec))
            self.log['te_recall'].append(np.mean(te_rec))
            self.log['tr_micro_f1'].append(np.mean(tr_f1))
            self.log['te_micro_f1'].append(np.mean(te_f1))

            print(f"{epoch}: "
                f"trloss:{np.mean(tr_loss):.4f} trP:{np.mean(tr_prec):.4f} trR:{np.mean(tr_rec):.4f} trF1:{np.mean(tr_f1):.4f} "
                f"teloss:{np.mean(te_loss):.4f} teP:{np.mean(te_prec):.4f} teR:{np.mean(te_rec):.4f} teF1:{np.mean(te_f1):.4f}")

            self.eval()

        if self.file_name is not None:
            pickle.dump(self.log, open(self.file_name + '.pkl', 'wb'))

    def predict(self, x, device='cpu'):
        self.to(device)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward_once(x.to(device))
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
    
class DL_input_data(Dataset):
    def __init__(self, windows, classes, cnn=False):
        self.cnn = cnn
        data, lengths = self.buffer(windows)
        self.data = data
        self.lengths = lengths

        # Multi Hot Labels: 

        class_indices = np.asarray(classes, dtype=int)
        N = class_indices.shape[0]
        out = np.zeros((N, 5), dtype=np.float32)

        for i, c in enumerate(class_indices):
            if c == 0:
                continue  # all zeros = null
            else:
                out[i, c-1] = 1.0  # shift down by 1 since we dropped null
                
        self.classes = torch.tensor(out, dtype=torch.float32)

    def buffer(self, input):
        if self.cnn:
            lengths = torch.tensor(np.array([len(w) for w in input]), dtype=torch.long)
            max_len = max(lengths).item()
            num_channels = input[0].shape[1]
            num_samples = input[0].shape[2] if len(input[0].shape) > 2 else 1 
            padded_emg = np.zeros((len(input), max_len, num_channels, num_samples))
            for i, e in enumerate(input):
                padded_emg[i, 0:e.shape[0], :, :] = e 
        else: 
            lengths = torch.tensor(np.array([len(w) for w in input]), dtype=torch.long)
            padded_emg = np.zeros((len(input), max(lengths).item(), input[0].shape[1])) 
            for i, e in enumerate(input):
                padded_emg[i, 0:e.shape[0], :] = e
        return torch.tensor(padded_emg, dtype=torch.float32), lengths

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        length = self.lengths[idx]
        return data, label, length

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(windows, classes, batch_size=512, cnn=False, shuffle=True):
    obj = DL_input_data(windows, classes, cnn)
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=shuffle) 
    return dl

def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False