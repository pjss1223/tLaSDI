class SparseAutoEncoder(nn.Module):
    """Sparse Autoencoder"""
    def __init__(self, layer_vec, activation):
        super(SparseAutoEncoder, self).__init__()
        self.layer_vec = layer_vec
        self.dim_latent = layer_vec[-1]
        self.activation = activation
        #self.activation_vec = ['linear'] + (len(self.layer_vec)-3)*[self.activation] + ['linear']
        self.activation_vec = ['linear'] + (len(self.layer_vec)-3)*[self.activation] + ['linear']


        # Encode
        self.steps = len(self.layer_vec)-1
        self.fc_encoder = nn.ModuleList()
        for k in range(self.steps):
            self.fc_encoder.append(nn.Linear(self.layer_vec[k], self.layer_vec[k+1]))

        # Decode
        self.fc_decoder = nn.ModuleList()
        for k in range(self.steps):
            self.fc_decoder.append(nn.Linear(self.layer_vec[self.steps - k], self.layer_vec[self.steps - k - 1]))

    def activation_function(self, x, activation):
        if activation == 'linear': x = x
        elif activation == 'sigmoid': x = torch.sigmoid(x)
        elif activation == 'relu': x = F.relu(x)
        elif activation == 'rrelu': x = F.rrelu(x)
        elif activation == 'tanh': x = torch.tanh(x)
        elif activation == 'elu': x = F.elu(x)
        else: raise NotImplementedError
        return x

    # Encoder
    def encode(self, x):
        idx = 0
        for layer in self.fc_encoder:
            #print(x.shape)
            x = layer(x)
            x = self.activation_function(x, self.activation_vec[idx])
            idx += 1
        return x

    # Decoder
    def decode(self, x):
        idx = 0
        for layer in self.fc_decoder:
            x = layer(x)
            x = self.activation_function(x, self.activation_vec[idx])
            idx += 1
        return x

    # Forward pass
    def forward(self, z):
        x = self.encode(z)
        z_reconst = self.decode(x)
        return z_reconst, x

    # Normalization
    def normalize(self, z):
        return z

    def denormalize(self, z_norm):
        return z_norm

