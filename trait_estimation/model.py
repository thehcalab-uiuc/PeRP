#!/usr/bin/env python

__author__ = "Aamir Hasan"
__version__ = "0.1"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""
For VAE implementation etc refer to: 
https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/1_VAE_mnist_sigmoid_mse.ipynb
"""

from typing import Tuple
import torch
import torch.nn as nn


class TraitVAE(nn.Module):
    """_The VAE model for Driver Trait Estimation._"""

    def __init__(self, config: dict):
        """_Initializes the model parameters and components._

        Args:
            config (dict): _Configs passes in by the trainer._
        """
        super(TraitVAE, self).__init__()
        self.encoder = nn.LSTM(
            config.state_dim,
            config.encoder_dim,
            config.n_encoder_layers,
            batch_first=True,
        )

        self.z_mean = nn.Linear(config.encoder_dim, config.latent_dim)
        self.z_log_var = nn.Linear(config.encoder_dim, config.latent_dim)

        self.decoder_rnn = nn.LSTM(
            config.latent_dim,
            config.decoder_dim,
            config.n_decoder_layers,
            batch_first=True,
        )
        self.decoder_mlp = nn.Linear(config.decoder_dim, config.state_dim)

        self.decoder_dim = config.decoder_dim
        self.latent_dim = config.latent_dim
        self.state_dim = config.state_dim
        self.n_encoder_layers = config.n_encoder_layers
        self.n_decoder_layers = config.n_decoder_layers

    def encode(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_Encodes the input data into the latent space by using the reparametrization trick._

        Args:
            data (torch.Tensor): _The data to be encoded. Tensor of shape (batch_size, trajectory length, state_dim)._

        Returns:
            _(torch.Tensor, torch.Tensor, torch.Tensor)_: _Tuple with the encoded latent vector, mean vector, and log variance vector for the data passed in._
        """
        # data -> (batch_size, traj_len, state_dim)
        encoded, _ = self.encoder(data)
        # encoded -> (batch_size, traj_len, latent_dim)
        encoded = encoded[:, -1, :]
        # encoded -> (batch_size, 1, latent_dim)
        z_mean, z_log_var = self.z_mean(encoded), self.z_log_var(encoded)
        # z_mean, z_log_var -> (batch_size, 1, latent_dim)
        encoded = self.reparameterize(z_mean, z_log_var)
        # encoded -> (batch_size, latent_dim)
        return encoded, z_mean, z_log_var

    def reparameterize(
        self, z_mean: torch.Tensor, z_log_var: torch.Tensor
    ) -> torch.Tensor:
        """_Reparametrizes the data as per VAE models._

        Args:
            z_mean (torch.Tensor): _mean vectors._
            z_log_var (torch.Tensor): _log variance vectors._

        Returns:
            torch.Tensor: _Reparameterized representation of the vectors._
        """
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(z_mean.device)
        z = z_mean + eps * torch.exp(z_log_var / 2.0)
        return z

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """_Processes the input data and outputs the latent representation and reconstructions of the input data. Also outputs the means and log variances so that trainer can calculate the KL divergence for the batch._

        Args:
            data (torch.Tensor): _Tensor of shape (batch_size, traj_len, state_dim) to be processed._

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _The latent representation of the data, the mean, the log variances, and the reconstruction of the input data._
        """
        batch_size, traj_len, _ = data.shape

        # encode the data
        # encoded -> (batch_size, latent_dim)
        encoded, z_mean, z_log_var = self.encode(data)

        # init the hidden and cell state for the decoder
        hidden_state = torch.zeros(
            self.n_decoder_layers, batch_size, self.decoder_dim, device=data.device
        )
        cell_state = torch.zeros(
            self.n_decoder_layers, batch_size, self.decoder_dim, device=data.device
        )

        # decode
        decoder_input = encoded.view(batch_size, 1, self.latent_dim).repeat(
            1, traj_len, 1
        )
        # decoder_input -> (batch_size, traj_len, latent_dim)
        decoder_output, _ = self.decoder_rnn(decoder_input, (hidden_state, cell_state))
        # decoder_output -> (batch_size, traj_len, decoder_dim)

        # go from the decoder output space back to the state space
        decoder_output = self.decoder_mlp(decoder_output.reshape(-1, self.decoder_dim))
        decoder_output = decoder_output.view(batch_size, traj_len, self.state_dim)
        # decoder_output -> (batch_size, traj_len, state_dim)

        return encoded, z_mean, z_log_var, decoder_output
