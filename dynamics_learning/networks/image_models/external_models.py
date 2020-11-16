import torch
from torch import nn as nn

# Modified from: https://github.com/utiasSTARS/robust-latent-srl/blob/master/srl/srl/train.py

# ----------------- #
# AUXILIARY CLASSES #
# ----------------- #


class ConvBlock(nn.Module):
    """Convolution block."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        bn: bool = False,
        drop: bool = False,
        nl: bool = True,
    ) -> None:
        """Initialize the conv block."""
        super(ConvBlock, self).__init__()
        self._conv = nn.Conv2d(
            c_in, c_out, kernel, stride=stride, padding=padding, bias=False
        )
        self._bn = nn.BatchNorm2d(c_out, track_running_stats=True) if bn else None
        self._drop = nn.Dropout(p=0.5) if drop else None
        self._nl = nn.ReLU() if nl else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self._conv(x)
        if self._bn is not None:
            out = self._bn(out)
        if self._drop is not None:
            out = self._drop(out)
        if self._nl is not None:
            out = self._nl(out)
        return out


class ConvTBlock(nn.Module):
    """Convolution transpose block."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        bn: bool = False,
        drop: bool = False,
        nl: bool = True,
    ) -> None:
        """Initialize the conv transpose block."""
        super(ConvTBlock, self).__init__()
        self._conv = nn.ConvTranspose2d(
            c_in, c_out, kernel, stride=stride, padding=padding, bias=False
        )
        self._bn = nn.BatchNorm2d(c_out, track_running_stats=True) if bn else None
        self._drop = nn.Dropout(p=0.5) if drop else None
        self._nl = nn.ReLU() if nl else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self._conv(x)
        if self._bn is not None:
            out = self._bn(out)
        if self._drop is not None:
            out = self._drop(out)
        if self._nl is not None:
            out = self._nl(out)
        return out


class FullyConvEncoderVAE(nn.Module):
    """Conv encoder."""

    def __init__(
        self,
        input=1,
        latent_size=12,
        bn=True,
        extra_scalars=0,
        extra_scalars_conc=0,
        drop=False,
        nl=nn.ReLU(),
        stochastic=True,
        img_dim="64",
    ) -> None:
        """Initialize encoder."""
        super(FullyConvEncoderVAE, self).__init__()
        self.stochastic = stochastic
        self.layers = nn.ModuleList()
        self.extra_scalars = extra_scalars
        self.extra_scalars_conc = extra_scalars_conc
        self.latent_size = latent_size

        self.layers.append(ConvBlock(input, 32, 4, stride=2, bn=bn, drop=drop, nl=True))
        self.layers.append(ConvBlock(32, 64, 4, stride=2, bn=bn, drop=drop, nl=True))

        if img_dim in ["128", "64", "32"]:
            self.layers.append(
                ConvBlock(64, 128, 4, stride=2, bn=bn, drop=drop, nl=True)
            )

        if img_dim in ["128", "64"]:
            self.layers.append(
                ConvBlock(128, 256, 4, stride=2, bn=bn, drop=drop, nl=True)
            )

        if img_dim == "64":
            n_size = 256 * 2 * 2
        elif img_dim == "128":
            n_size = 256 * 6 * 6
        elif img_dim == "32":
            n_size = 128 * 2 * 2
        elif img_dim == "16":
            n_size = 64 * 2 * 2
        else:
            raise NotImplementedError()

        if self.stochastic:
            self.fc_mu = nn.Linear(n_size, latent_size + extra_scalars_conc)
            self.fc_logvar = nn.Linear(n_size, latent_size)
        else:
            self.fc = nn.Linear(n_size, latent_size)

        if self.extra_scalars > 0:
            self.fc_extra = nn.Sequential(
                nn.Linear(n_size, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.extra_scalars),
                nn.ELU(alpha=4),
            )
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Forward pass."""
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.flatten(x)
        if self.stochastic:
            x_mu = self.fc_mu(x)
            mu = x_mu[:, : self.latent_size]
            logvar = self.fc_logvar(x)
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # UNUSED CODE FROM ORIGINAL REPO
            ############################################################################
            # Extra variables with shared network
            # if self.extra_scalars > 0:
            #     extra_scalars = self.fc_extra(x)
            #     # return z, mu, logvar, torch.exp(extra_scalars)
            #     return z, mu, logvar, extra_scalars

            # if self.extra_scalars_conc > 0:
            #     extra_scalars = x_mu[:, self.latent_size:]
            #     return z, mu, logvar, extra_scalars

            # return z, mu, logvar
            ############################################################################

            return torch.cat([mu, logvar], dim=-1)
        else:
            z = self.fc(x)
            if self.extra_scalar_size > 0:
                extra_scalars = self.fc_extra(x)
                return z, extra_scalars

            if self.extra_scalars_conc > 0:
                extra_scalars = x_mu[self.latent_size :]
                return z, extra_scalars
            return z


class FullyConvDecoderVAE(nn.Module):
    """Conv Decoder."""

    def __init__(
        self,
        input=1,
        latent_size=12,
        output_nl=nn.Tanh(),
        bn=True,
        drop=False,
        nl=nn.ReLU(),
        img_dim="64",
    ) -> None:
        """Initialize the decoder."""
        super(FullyConvDecoderVAE, self).__init__()
        self.bn = bn
        self.drop = drop
        self.layers = nn.ModuleList()

        if img_dim == "64":
            n_size = 256 * 2 * 2
        elif img_dim == "128":
            n_size = 256 * 6 * 6
        elif img_dim == "32":
            n_size = 128 * 2 * 2
        elif img_dim == "16":
            n_size = 64 * 2 * 2
        else:
            raise NotImplementedError()

        self.layers.append(
            ConvTBlock(n_size, 128, 5, stride=2, bn=bn, drop=drop, nl=True)
        )

        if img_dim == "64":
            self.layers.append(
                ConvTBlock(128, 64, 5, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(64, 32, 6, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(32, input, 6, stride=2, bn=bn, drop=drop, nl=False)
            )
        elif img_dim == "128":
            self.layers.append(
                ConvTBlock(128, 64, 5, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(64, 32, 5, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(32, 16, 6, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(16, input, 6, stride=2, bn=bn, drop=drop, nl=False)
            )
        elif img_dim == "32":
            self.layers.append(
                ConvTBlock(128, 64, 5, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(64, 32, 6, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(32, 16, 6, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvBlock(16, input, 3, stride=2, padding=1, bn=bn, drop=drop, nl=False)
            )
        elif img_dim == "16":
            self.layers.append(
                ConvTBlock(128, 64, 5, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(64, 32, 6, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvTBlock(32, 16, 6, stride=2, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvBlock(16, 8, 3, stride=2, padding=1, bn=bn, drop=drop, nl=True)
            )
            self.layers.append(
                ConvBlock(8, input, 3, stride=2, padding=1, bn=bn, drop=drop, nl=False)
            )
        else:
            raise NotImplementedError()

        if output_nl is not None:
            self.layers.append(output_nl)

        self.linear = nn.Linear(latent_size, n_size, bias=False)
        self.batchn = nn.BatchNorm1d(n_size)
        self.dropout = nn.Dropout(p=0.5)
        self.nl = nl

    def forward(self, x):
        """Forward pass."""
        if self.bn:
            x = self.nl(self.batchn(self.linear(x)))
        elif self.drop:
            x = self.nl(self.dropout(self.linear(x)))
        else:
            x = self.nl(self.linear(x))

        x = x.unsqueeze(-1).unsqueeze(-1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x
