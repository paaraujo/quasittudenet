
import torch
import torch.nn as nn
   
class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_activation=True):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        if last_activation:
            self.net = nn.Sequential(self.linear, self.batchnorm, self.gelu)
        else:
            self.net = self.linear

    def forward(self, x):
        out = self.net(x)
        return out
    

class LinearEstimator(nn.Module):
    def __init__(self, in_channels, num_channels) -> None:
        super(LinearEstimator, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = in_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [LinearBlock(in_channels, out_channels, False)]
            else:
                layers += [LinearBlock(in_channels, out_channels, True)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, encoder_params) -> None:
        super(Encoder, self).__init__()
        self.encoder_params = encoder_params
        layers = []
        width = len(encoder_params['channels'])
        for i in range(width):
            in_channels = encoder_params['in_channels'] if i == 0 else encoder_params['channels'][i-1]
            layers += [self._conv(in_channels, encoder_params['channels'][i], encoder_params['kernel_size'], encoder_params['dilation'], encoder_params['stride'], encoder_params['padding'], encoder_params['groups'])]
        self.net = nn.Sequential(*layers)

    def _conv(self, in_channels, out_channels, kernel_size, dilation, stride, padding, groups):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        ) 

    def forward(self, x):
        out = self.net(x)
        return out


class Decoder(nn.Module):
    def __init__(self, decoder_params) -> None:
        super(Decoder, self).__init__()
        layers = []
        width = len(decoder_params['channels'])
        for i in range(width):
            in_channels = decoder_params['in_channels'] if i == 0 else decoder_params['channels'][i-1]
            layers += [self._convtranspose(in_channels, decoder_params['channels'][i], decoder_params['kernel_size'], decoder_params['dilation'], decoder_params['stride'], decoder_params['padding'], decoder_params['groups'])]
        self.net = nn.Sequential(*layers)

    def _convtranspose(self, in_channels, out_channels, kernel_size, dilation, stride, padding, groups):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        out = self.net(x)
        return out
    

class AttitudeEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttitudeEstimator, self).__init__()
        self.network = nn.Sequential(nn.Flatten(), LinearEstimator(input_size, output_size))

    def forward(self, x):
        return self.network(x)


class QuasittudeNet(nn.Module):

    def __init__(self, encoder_params, decoder_params, estimator_params, misalignment) -> None:
        super(QuasittudeNet, self).__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)
        self.latent_space_shape = torch.tensor(self._get_output_shape(self.encoder, estimator_params['input_size']), dtype=torch.int32)
        self.estimator = AttitudeEstimator(torch.prod(self.latent_space_shape, dtype=torch.int32), estimator_params['channels'])
        self.misalignment = torch.nn.Parameter(torch.tensor([misalignment]), requires_grad=True) 

    def forward(self, x):
        # Sending data to the encoder
        latent_space = self.encoder(x)
        # Decoding data
        decoded = self.decoder(latent_space)
        # Estimating roll and pitch using latent space representation
        rp = self.estimator(latent_space)
        if not self.training:
            rp += self.misalignment.expand(rp.size(0), -1)
        return decoded, rp

    def loss(self, raw, decoded, rp_latent, rotated_g):
        # Reconstruction loss
        loss_rec = torch.nn.functional.mse_loss(raw, decoded)
        # Attitude loss
        g = torch.tensor([0., 0., 1.], dtype=torch.float, device=rp_latent.device)
        rot_rp_latent = self._batch_euler_to_rotmat(rp_latent + self.misalignment.expand(rp_latent.size(0), -1))
        estimated_rotated_g_latent = torch.einsum('bji, j -> bi', rot_rp_latent, g)
        loss_att_latent = torch.nn.functional.mse_loss(estimated_rotated_g_latent, rotated_g, reduction='mean')
        # Getting total loss
        beta = 100
        total_loss = loss_rec + beta*loss_att_latent
        return total_loss

    def _batch_euler_to_rotmat(self, angles):
        """
        Convert batched Euler angles to 3D rotation matrices, assuming z rotation is 0.
        :param angles: Tensor of shape [batch_size, 2, seq_len] containing Euler angles.
                    angles[:, 0, :] are rotations around the x-axis,
                    angles[:, 1, :] are rotations around the y-axis.
        :return: Tensor of shape [batch_size, seq_len, 3, 3] containing 3D rotation matrices.
        """
        # Extract angles: assume angles are in radians
        angles_x, angles_y = angles[:, 0], angles[:, 1]
        
        # Precompute cosine and sine of angles
        cos_x = torch.cos(angles_x)
        sin_x = torch.sin(angles_x)
        cos_y = torch.cos(angles_y)
        sin_y = torch.sin(angles_y)
        
        # Initialize rotation matrices
        R_x = torch.zeros(angles.size(0), 3, 3, device=angles.device)
        R_y = torch.zeros_like(R_x)
        
        # Rotation matrix for rotation around x-axis
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos_x
        R_x[:, 1, 2] = -sin_x
        R_x[:, 2, 1] = sin_x
        R_x[:, 2, 2] = cos_x
        
        # Rotation matrix for rotation around y-axis
        R_y[:, 0, 0] = cos_y
        R_y[:, 0, 2] = sin_y
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin_y
        R_y[:, 2, 2] = cos_y
        
        # Compute final rotation matrix by multiplying R_y and R_x for each sequence
        # Note: Matrix multiplication is batched across the first dimension (batch size) and the second dimension (sequence length)
        R = torch.matmul(R_y, R_x)
        
        return R

    def _get_output_shape(self, model, tensor_dim):
        return model(torch.rand(*(tensor_dim))).data.shape
