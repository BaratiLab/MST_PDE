# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import torch
from torch import nn
import numpy as np


class LayerNorm2d(nn.Module):
    """
    Just a LayerNorm on the feature channel for 2D data with shape (B, C, H, W)
    """
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.LayerNorm(dim)

    def forward(self, x):
        # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        # (B, H, W, C)
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)
        # (B, C, H, W)
        return x


class Mlp(nn.Module):
    """
    2 layer FC pointwise network for 2D data using 1x1 convolutions
    Architecture:
        1x1 conv - activation - dropout(optional) - 1x1 conv
    """
    def __init__(self,
                 in_dim,
                 hidden_dim = None,
                 out_dim = None,
                 act = nn.LeakyReLU,
                 drop = 0
                 ):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.act = act()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def sin_pos_encoding(x):
    """
    absolute sinisidual positional encoding with shape (nx, ny, dim)
    """
    nx, ny = x.shape[-2:]
    xfreqs = 1./torch.arange(2, nx+1)
    yfreqs = 1./torch.arange(2, ny+1)
    x_mesh, y_mesh = torch.meshgrid(torch.arange(1, nx+1), torch.arange(1, ny+1))
    x_mesh, y_mesh = x_mesh[..., None], y_mesh[..., None]
    sinx = torch.sin( 2*np.pi*xfreqs[1:] *x_mesh)
    cosx = torch.cos( 2*np.pi*xfreqs *x_mesh)
    siny = torch.sin( 2*np.pi*yfreqs[1:] *y_mesh)
    cosy = torch.cos( 2*np.pi*yfreqs *y_mesh)
    features = torch.cat([sinx, cosx, siny, cosy], dim=-1)
    return features


class Functional_pos_embedding(nn.Module):
    """
    2 layer network to learn absolute positional embedding from positional features
    if peroidic is False, features will be x and y
    if peroidic is True , features will be sines and cosines
    Architecture:
        linear - LeakyReLU - linear

    The output shape is (1, embed_dim, nx, ny)
    """
    def __init__(self, embed_dim=64, hidden_dim=128, periodic=True):
        super().__init__()
        self.periodic = periodic
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim)
            )

    def pos_features(self, x):
        """
        Returns
        -------
        features: torch tensor of shape (nx, ny, 2) containing x and y
        """
        nx, ny = x.shape[-2:]
        x_mesh, y_mesh = torch.linspace(0, 1, nx+1)[1:], torch.linspace(0, 1, ny+1)[1:]
        x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh) # (nx, ny)
        return torch.stack([x_mesh, y_mesh], dim=-1) # (nx, ny, 2)

    def pos_features_periodic(self, x):
        """
        Returns
        -------
        features: torch tensor of shape (nx, ny, d)
        """
        return sin_pos_encoding(x) # (nx, ny, d0)

    def forward(self, x):
        if self.periodic:
            pos_features = self.pos_features_periodic(x)
        else:
            pos_features = self.pos_features(x)
        # (1, Nq, Nk, d) --> (1, Nq, Nk, embed_dim)
        out = self.net(pos_features[None, ...].to(x.device))
        return out.permute(0, 3, 1, 2) # (1, embed_dim, Nq, Nk)


class Functional_rel_pos_embedding(nn.Module):
    """
    2 layer network to learn relative positional embedding from positional features
    if r_only is True, features is merely the radius from query to key (r)
    if r_only is False, features are [r, theta, x, y, |x|, |y|]
    if peroidic is True, features will e periodic sines and cosines
    Architecture:
        linear - LeakyReLU - linear

    THe output shape is (1, num_heads, Nq, Nk)
    """
    def __init__(self, num_heads=8, hidden_dim=64, r_only=True, periodic=False):
        super().__init__()
        self.periodic = periodic
        self.r_only = r_only
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_heads)
            )

    def pos_features(self, x):
        """
        Returns
        -------
        features: torch tensor of shape (Nq, Nk, 1 or 6)
        for [r] or [r, theta, x, y, |x|, |y|] respectively.
        Nq = Nk = nx*ny
        """
        nx, ny = x.shape[-2:]
        x_mesh, y_mesh = torch.linspace(0, 1, nx+1)[1:], torch.linspace(0, 1, ny+1)[1:]
        x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh) # (nx, ny)
        x_mesh, y_mesh = x_mesh.flatten(), y_mesh.flatten()
        dx = x_mesh[None, :] - x_mesh[:, None]
        dy = y_mesh[None, :] - y_mesh[:, None]
        r = (dx**2 + dy**2)**0.5

        if self.r_only:
            return r[..., None]

        theta = torch.atan2(dy, dx)
        return torch.stack([r, theta, dx, dy, dx.abs(), dy.abs()], dim=-1)

    def pos_features_periodic(self, x):
        """
        Returns
        -------
        features: torch tensor of shape (Nq, Nk, d)
        Nq = Nk = nx*ny
        """
        pos_info = sin_pos_encoding(x)
        pos_info = pos_info.flatten(0, 1)  # (Nq*d)
        dot = torch.einsum('nd,md->nm', pos_info, pos_info)[..., None]
        diff = pos_info[:, None, :] - pos_info[None, :, :]
        return torch.cat([dot, diff, diff.abs()], dim=-1)

    def forward(self, x):

        if self.periodic:
            pos_features = self.pos_features_periodic(x)
        else:
            pos_features = self.pos_features(x)
        out = self.net(pos_features[None, ...].to(x.device))
        out = out.permute(0, 3, 1, 2)
        return out


class Learnable_rel_pos_embedding(nn.Module):
    """
    A learnable relative positional embedding, with specified symmetric property

    Here, 00 is the query position, and the surroundings are keys.
    Anything further than that is treated as padding idx.
    ---------------------------------------------------------------------------
    HALF symmetric:
    (Symmetric with respect to x-axis and y-axis)
    It tokenizes the surrounding key positions like this:
    15 14 13 12 13 14 15
    11 10 09 08 09 10 11
    07 06 05 04 05 06 07
    03 02 01 00 01 02 03
    07 06 05 04 05 06 07
    11 10 09 08 09 10 11
    15 14 13 12 13 14 15
    ---------------------------------------------------------------------------
    FULL symmetric:
    (Symmetric with respect to x-axis, y-axis, x=y, x=-y)
    It tokenizes the surrounding key positions like this:
    09 08 07 06 07 08 09
    08 05 04 03 04 05 08
    07 04 02 01 02 04 07
    06 03 01 00 01 03 06
    07 04 02 01 02 04 07
    08 05 04 03 04 05 08
    09 08 07 06 07 08 09
    ---------------------------------------------------------------------------
    Then, nn.Embedding is used to map each token to a
    learnable embedding vector of the dimension specified.

    The output has shape (1, num_heads, Nq, Nk) where Nq = Nk = nx*ny.
    """
    def __init__(self, rng=5, num_heads=8, sym='half'):
        super().__init__()
        self.rng = rng
        self.sym = sym
        if sym == 'half':
            self.max_idx = (rng+1)**2
        elif sym == 'full':
            self.max_idx = (rng+1)*(rng+2)//2
        self.embedder = nn.Embedding(num_embeddings = self.max_idx+1,
                                     embedding_dim = num_heads,
                                     padding_idx = self.max_idx)

    def tokenize(self, x):
        nx, ny = x.shape[-2:]
        x_mesh, y_mesh = torch.arange(nx), torch.arange(ny)
        x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh) # (nx, ny)
        x_mesh, y_mesh = x_mesh.flatten(), y_mesh.flatten()
        dx = x_mesh[None, :] - x_mesh[:, None]
        dy = y_mesh[None, :] - y_mesh[:, None]
        if self.sym == 'half':
            tokens = dx.abs()*(self.rng+1) + dy.abs()
            tokens[torch.logical_or(dx.abs()>self.rng, dy.abs()>self.rng)] = self.max_idx
        elif self.sym == 'full':
            rel_pos = torch.stack([dx,dy])
            l = rel_pos.abs().max(dim=0)[0]
            tokens = rel_pos.abs().sum(dim=0) + l*(l-1)/2
            tokens = torch.clip(tokens, max=self.max_idx)
        return tokens.to(torch.int)

    def forward(self, x):
        rel_pos_token = self.tokenize(x)
        rel_pos_embedding = self.embedder(rel_pos_token.to(x.device))
        return rel_pos_embedding.permute([2, 0, 1])


class Attention2D(nn.Module):
    def __init__(self, # 12 args (11 if out_dim is None)
                 dim,
                 out_dim = None,
                 num_heads = 8,
                 split = False,

                 attn_method = 1,
                 softmax = True,
                 pos_type = 1,
                 attn_rng = 3, # for pos_type 4 or 5 (learnable embedding)

                 qkv_bias = True,
                 qk_scale = None,
                 attn_drop = 0,
                 proj_drop = 0,
                 ):
        """
        A flexible 2D attention module (based on ViT)
        attn_method options:
            0: no positional encoding
                qkv = QKV(x),    attn = [softmax](qk/d**0.5)
            1: classical way to use positional encoding (add to input at first)
                qkv = QKV(x+pe), attn = [softmax](qk/d**0.5)
            2: positional info added to qk after qk is calculated
                qkv = QKV(x),    attn = [softmax](pe + qk/d**0.5)
            3: positional info multiplied to attention weights
                qkv = QKV(x),    attn = [softmax](pe * qk/d**0.5)
            4: positional info added to transformed qk
                qkv = QKV(x),    attn = pe + [softmax](qk/d**0.5)
            5: positional info multiplied to transformed qk
                qkv = QKV(x),    attn = pe * [softmax](qk/d**0.5)
            6: No attention or token mixing (return 0)

            *** IF NO SOFTMAX, 2 = 4 , and 3 = 5 as well!

        pos_type options:

            0: absolute NN(x, y) in shape (2, nx, ny)
            1: absolute NN(sinisidual) in shape (d, nx, ny) to be added to x

            The following are of the shape (num_heads, Nq, Nk) and relative
            so they are to be used after qk is calculated:

            2: relative cartesian pos embedding NN(r)
            3: relative cartesian pos embedding NN(r, theta, dx, dy, |dx|, |dy|)
            4: relative periodic pos embedding NN(periodic features)
            5: half symmetric learnable relative pos embedding
            6: symmetric learnable relative pos embedding
        """
        super().__init__()
        self.attn_method = attn_method
        if attn_method == 6:
            return

        self.softmax = softmax
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        out_dim = out_dim or dim

        self.Q = nn.Conv2d(dim, dim, 1, bias=qkv_bias,
                           groups = num_heads if split else 1)
        self.K = nn.Conv2d(dim, dim, 1, bias=qkv_bias,
                           groups = num_heads if split else 1)
        self.V = nn.Conv2d(dim, dim, 1, bias=qkv_bias,
                           groups = num_heads if split else 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        #
        if attn_method == 0:
            return # no positional embedding needed!
        assert not (attn_method in [1] and pos_type not in [0,1]), 'Choose an absolute pos_type'
        assert not (attn_method in [2,3,4,5] and pos_type not in [2,3,4,5,6]), 'Choose a relative pos_type'
        self.pos_type = pos_type
        if pos_type == 0:
            self.pos_embedder = Functional_pos_embedding(embed_dim=dim, periodic=False)
        elif pos_type == 1:
            self.pos_embedder = Functional_pos_embedding(embed_dim=dim)
        elif pos_type == 2:
            self.pos_embedder = Functional_rel_pos_embedding(num_heads=num_heads)
        elif pos_type == 3:
            self.pos_embedder = Functional_rel_pos_embedding(num_heads=num_heads, r_only=False)
        elif pos_type == 4:
            self.pos_embedder = Functional_rel_pos_embedding(num_heads=num_heads, periodic=True)
        elif pos_type == 5:
            self.pos_embedder = Learnable_rel_pos_embedding(num_heads=num_heads, sym='half', rng=attn_rng)
        elif pos_type == 6:
            self.pos_embedder = Learnable_rel_pos_embedding(num_heads=num_heads, sym='full', rng=attn_rng)

    def forward(self, x):
        if self.attn_method == 6:
            return 0.

        B, C, H, W = x.shape
        if self.attn_method == 1:
            x = x + self.pos_embedder(x)
        q, k, v = self.Q(x), self.K(x), self.V(x)

        attn = torch.einsum('bhdn,bhdm->bhnm',
                            q.reshape(B, self.num_heads, self.head_dim, -1),
                            k.reshape(B, self.num_heads, self.head_dim, -1)
                            ) * self.scale

        if self.attn_method in [2,3,4,5]:
            pos_embedding = self.pos_embedder(x)

        if self.attn_method == 2:
            attn = pos_embedding + attn
        elif self.attn_method == 3:
            attn = pos_embedding * attn

        if self.softmax:
            attn = attn.softmax(dim=-1)

        if self.attn_method == 4:
            attn = pos_embedding + attn
        elif self.attn_method == 5:
            attn = pos_embedding * attn

        attn = self.attn_drop(attn)
        x = torch.einsum('bhnm,bhdm->bhdn',
                         attn,
                         v.reshape(B, self.num_heads, self.head_dim, -1)
                         ).flatten(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CNN_Encoder(nn.Module):
    def __init__(self,
                 channels = [1, 8, 16, 32, 64],
                 act = nn.LeakyReLU,
                 padding_mode = 'zeros'
                 ):
        super().__init__()
        n = len(channels) - 1
        layers = []
        for i in range(n):
            layers += [
                nn.Conv2d(in_channels = channels[i],
                          out_channels = channels[i+1],
                          kernel_size = 3,
                          stride = 1,
                          padding = 'same',
                          padding_mode = padding_mode
                          ),
                nn.AvgPool2d(2),
                act()
                ]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)


class CNN_Decoder(nn.Module):
    def __init__(self,
                 channels = [64, 32, 16, 8, 1],
                 act = nn.LeakyReLU,
                 padding_mode = 'zeros',
                 upsample_mode = 'bilinear'
                 ):
        super().__init__()
        n = len(channels) - 1
        layers = []
        for i in range(n):
            layers += [
                nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
                nn.Conv2d(in_channels = channels[i],
                          out_channels = channels[i+1],
                          kernel_size = 3,
                          stride = 1,
                          padding = 'same',
                          padding_mode = padding_mode
                          ),
                act()
                ]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)


class Transformer2D(nn.Module):
    """
    A flexible multiLayer transformer encoder.
    Arguments are self-explanatory.
    The feature dimension, however, is expanded in
    the concat layer of the attention block.
    For different attention methods, refer to Attention2D module.
    """
    def __init__(self,
                 dim = 128,
                 num_heads = [8, 8, 8, 8],
                 split = False,

                 attn_method = 1,
                 softmax = True,
                 pos_type = 1,
                 attn_rng = 3,

                 qkv_bias = True,
                 qk_scale = None,
                 attn_drop = 0,
                 proj_drop = 0,

                 use_norm = False,
                 use_fc = True
                 ):
        super().__init__()
        self.n_layers = len(num_heads)

        self.attn_norms = nn.ModuleList([
            LayerNorm2d(dim) if use_norm else nn.Identity()
            for i in range(self.n_layers)
            ])
        self.attn_layers = nn.ModuleList([
            Attention2D(
                dim,
                num_heads = num_heads[i],
                split = split,

                attn_method = attn_method,
                softmax = softmax,
                pos_type = pos_type,
                attn_rng = attn_rng, # for pos_type 4 or 5 (learnable embedding)

                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
                attn_drop = attn_drop,
                proj_drop = proj_drop
                )
            for i in range(self.n_layers)
            ])

        self.use_fc = use_fc
        if use_fc:
            self.fc_norms = nn.ModuleList([
                LayerNorm2d(dim) if use_norm else nn.Identity()
                for i in range(self.n_layers)
                ])
            self.fc_layers = nn.ModuleList([
                Mlp(in_dim = dim, hidden_dim = 4*dim)
                for i in range(self.n_layers)
                ])

    def forward(self, x):
        for i in range(self.n_layers):
            x = x + self.attn_layers[i](self.attn_norms[i](x))
            if self.use_fc:
                x = x + self.fc_layers[i](self.fc_norms[i](x))
        return x


def L2normLoss(pred, true, dim=(-1,-2,-3), mean=True):
    """
    Parameters
    ----------
    pred : torch tensor of shape (B, C or T, H, W)
    true : torch tensor of shape (B, C or T, H, W)
    It can also not have the B dimenstion

    Returns
    -------
    loss : scalar
        normalized L2 loss
    """
    pred = torch.as_tensor(pred, dtype=torch.float)
    true = torch.as_tensor(true, dtype=torch.float)
    error = pred - true
    error_norm = torch.norm(error, p=2, dim=dim)
    x_norm = torch.norm(true, p=2, dim=dim)
    batched_loss = error_norm / x_norm
    if mean:
        return batched_loss.mean()
    return batched_loss
