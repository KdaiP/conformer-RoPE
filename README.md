<div align="center">

# conformer-RoPE

_A refined Conformer block with Rotary Position Embedding, modified from <a href="https://github.com/lucidrains/conformer">lucidrains' implement</a>_

</div>

## Modification:

1. Use [Rotary Position Embedding](https://nn.labml.ai/transformers/rope/index.html) instead of relative embedding

2. Use pytorch official implement of [GLU](https://pytorch.org/docs/stable/generated/torch.nn.GLU) and [Swish](https://pytorch.org/docs/stable/generated/torch.nn.SiLU) activation, which are slightly faster

3. Use pytorch official implement of [scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), which can automatically switch to flash attention or xformers if possible

4. Remove the the dependency of [einops](https://github.com/arogozhnikov/einops), now we only need pytorch

## Usage

```python
import torch
from conformer import Conformer

model = Conformer(n_layers=3, 
                  hidden_channels=192, 
                  filter_channels=768, 
                  n_heads=2, 
                  kernel_size=3)

x = torch.randn([32, 192, 35]) # input shape: [batch_size, hidden_channels, time]
model(x) # (32, 192, 35)
```