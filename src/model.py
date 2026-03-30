import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features, 
            hidden_size=embedding_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=embedding_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        # Encode
        _, (hidden_n, _) = self.encoder(x)
        encoded = hidden_n[-1, :, :] 
        
        # Repeat context vector
        repeated = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(repeated)
        reconstructed = self.output_layer(decoded)
        
        return reconstructed