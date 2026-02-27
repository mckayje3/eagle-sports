"""
College Football Models using Deep Framework
Uses the core deep learning module from documents/coding/deep
"""

import torch
import torch.nn as nn
from core.models import BaseTimeSeriesModel


class CFBFeedForwardModel(BaseTimeSeriesModel):
    """
    Feedforward neural network for college football predictions

    This model extends the deep framework's BaseTimeSeriesModel but
    uses a simple feedforward architecture suitable for tabular game data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
        task_type: str = 'classification',
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification or regression)
            num_layers: Number of hidden layers
            dropout: Dropout rate
            task_type: 'classification' or 'regression'
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs)

        self.task_type = task_type

        # Build feedforward layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(num_layers - 1):
            # Reduce dimension progressively
            next_dim = hidden_dim // (2 ** (i + 1)) if i < num_layers - 2 else hidden_dim // 2
            next_dim = max(next_dim, 16)  # Minimum 16 units

            layers.append(nn.Linear(hidden_dim if i == 0 else layers[-3].out_features, next_dim))
            layers.append(nn.ReLU())
            if i < num_layers - 2:  # No dropout on last hidden layer
                layers.append(nn.Dropout(dropout * 0.7))  # Reduced dropout in deeper layers

        # Output layer
        layers.append(nn.Linear(layers[-2].out_features, output_dim))

        # Add sigmoid for classification
        if task_type == 'classification':
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_dim)
               or (batch_size, sequence_length, input_dim) for compatibility

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Handle both tabular and sequence inputs
        if len(x.shape) == 3:
            # If sequence input, take the last timestep
            x = x[:, -1, :]

        return self.network(x)

    def get_config(self) -> dict:
        """Get model configuration"""
        config = super().get_config()
        config['task_type'] = self.task_type
        return config


class CFBLSTMModel(BaseTimeSeriesModel):
    """
    LSTM model for college football predictions using historical sequences

    This model uses team performance over multiple games to predict outcomes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        task_type: str = 'classification',
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features per game
            hidden_dim: LSTM hidden dimension
            output_dim: Output dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            task_type: 'classification' or 'regression'
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs)

        self.task_type = task_type

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Output activation
        if task_type == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state
        last_hidden = hidden[-1]

        # Fully connected layers
        output = self.fc(last_hidden)
        output = self.output_activation(output)

        return output

    def get_config(self) -> dict:
        """Get model configuration"""
        config = super().get_config()
        config['task_type'] = self.task_type
        return config
