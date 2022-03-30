import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self, context_size, dilation, in_dim, out_dim):
        super(TDNN, self).__init__()

        self.context_size = context_size
        self.dilation = dilation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_context_size = context_size * dilation - dilation + 1
        self._input_x_output = f"{in_dim * context_size} x {out_dim}"

        self.kernel = nn.Linear(in_dim * context_size, out_dim)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        assert len(x.shape) == 3
        B, T, D = x.shape
        assert (
            D == self.in_dim
        ), f"[error] Expected input dimension is {self.in_dim}, not {D}."

        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.in_dim),
            stride=(1, self.in_dim),
            dilation=(self.dilation, 1),
        )

        # x.shape: (N, output_dim*context_size, new_t)
        x = x.transpose(1, 2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        return x
