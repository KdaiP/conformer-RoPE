import torch
from torch import nn
import torch.nn.functional as F

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        return self.fn(x, **kwargs)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        
        if (kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")

        self.net = nn.Sequential(
            Transpose(),
            nn.LayerNorm(dim),
            Transpose(),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding =(kernel_size - 1) // 2),
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size, conv_expansion_factor=2, conv_kernel_size=31, p_dropout=0.):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.ff1 = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout)
        self.conv = ConformerConvModule(dim = hidden_channels, expansion_factor = conv_expansion_factor, kernel_size=conv_kernel_size, dropout=p_dropout)
        self.ff2 = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)

        self.attn = PreNorm(hidden_channels, self.attn)
        self.ff1 = Scale(0.5, PreNorm(hidden_channels, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(hidden_channels, self.ff2))

        self.post_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, x_mask=None):
        if x_mask is None:
            x_mask = torch.ones((x.size(0), 1, x.size(2)), device=x.device) # [b, 1, t]
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = self.ff1(x, x_mask=x_mask) + x
        x = self.attn(x, c=x, attn_mask=attn_mask) + x
        x = self.conv(x) + x
        x = self.ff2(x, x_mask=x_mask) + x
        x = self.post_norm(x.transpose(1,2)).transpose(1,2)
        return x

# Conformer

class Conformer(nn.Module):
    def __init__(self, n_layers, hidden_channels, filter_channels, n_heads, kernel_size, conv_expansion_factor=2, conv_kernel_size=31, p_dropout=0.):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(ConformerBlock(hidden_channels, filter_channels, n_heads, kernel_size, conv_expansion_factor, conv_kernel_size, p_dropout))

    def forward(self, x, x_mask=None):
        if x_mask is None:
            x_mask = torch.ones((x.size(0), 1, x.size(2)), device=x.device) # [b, 1, t]
        for block in self.layers:
            x = block(x, x_mask)
        return x
  
class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.k_channels = channels // n_heads
    self.conv_q = torch.nn.Conv1d(channels, channels, 1)
    self.conv_k = torch.nn.Conv1d(channels, channels, 1)
    self.conv_v = torch.nn.Conv1d(channels, channels, 1)

    # from https://nn.labml.ai/transformers/rope/index.html
    self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
    self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

    self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
    self.drop = torch.nn.Dropout(p_dropout)

    torch.nn.init.xavier_uniform_(self.conv_q.weight)
    torch.nn.init.xavier_uniform_(self.conv_k.weight)
    torch.nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)

  def forward(self, x, c, attn_mask=None):
      q = self.conv_q(x)
      k = self.conv_k(c)
      v = self.conv_v(c)

      x = self.attention(q, k, v, mask=attn_mask)

      x = self.conv_o(x)
      return x

  def attention(self, query, key, value, mask=None):
      b, d, t_s, t_t = (*key.size(), query.size(2))
      query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
      key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
      value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

      query = self.query_rotary_pe(query) # [b, n_head, t, c // n_head]
      key = self.key_rotary_pe(key)

      output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.p_dropout)
      output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
      return output
    
class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
    self.drop = nn.Dropout(p_dropout)
    self.act1 = nn.SiLU(inplace=True)

  def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
    
class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        x = x.permute(2, 0, 1, 3) # b h t d -> t b h d

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return torch.cat((x_rope, x_pass), dim=-1).permute(1, 2, 0, 3) # t b h d -> b h t d

class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)