import torch

from avalanche.models.base_model import BaseModel


class LSTM(torch.nn.Module, BaseModel):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        rnn_layers=1,
        batch_first=True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=batch_first,
        )
        self.classifier = torch.nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        # print()
        # print("(1) x.shape", x.shape)
        # print()
        # x = x.contiguous()
        # x = x.view(x.size(0), self._input_size)
        # print()
        # print("(2) x.shape", x.shape)
        # print()
        x, _ = self.rnn(x)
        # print()
        # print("(3) x.shape", x.shape)
        # print()
        x = x[:, -1] if self.batch_first else x[-1]
        # print()
        # print("(4) x.shape", x.shape)
        # print()
        x = self.classifier(x)
        # print()
        # print("(5) x.shape", x.shape)
        # print()
        return x

    def get_features(self, x):
        """
        Get features from model given input
        """
        x, _ = self.rnn(x)
        x = x[:, -1] if self.batch_first else x[-1]
        return x


__all__ = ["LSTM"]
