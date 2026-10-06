
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    
    The ordering of channels matters because ConvNeXt executes most of its normalization 
    and linear layers in the channels_last (BHWC) format.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; All in BCHW
    (2) DwConv -> Permute to BHWC -> LayerNorm -> Linear -> GELU -> Linear; Permute back
    We use implementation (2) as it matches the official version and runs faster in PyTorch.
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # Pointwise inverted bottleneck layers
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # LayerScale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        # each block has its own drop module and probability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # point wise? each neuron in the 4x expansion layer is connected to all neurons in the input layer, so it is a fully connected layer.
        x = self.pwconv1(x)
        x = self.act(x) # bottle neck only applies activation.
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        # input is residual connection, drop_path is applied to the output of the block before adding to input
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """Standalone ConvNeXt V1 network."""
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 6, 2], dims=[36, 96, 164], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem + 3 downsampling layers
        
        # Stem downsample layer: Conv 4x4 with stride 4
        # Stem downsample, non-overlapping convolution, reduces the spatial resolution of the input image by a factor of 4.
        # convolution for downsampling, followed by LayerNorm to normalize the feature maps.
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Intermediate downsample layers: Conv 2x2 with stride 2
        # The following only halve the size!!
        for i in range(2):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages
        # dp rate, linearly increases
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        cur = 0
        for i in range(3):
            # four stages, similar to the ResNet model.
            # a block is a basic ConvNeXt block, each stage has multiple blocks.
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final normalization layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnext_tiny(num_classes=1000, pretrained=False, **kwargs):
    """Constructs a ConvNeXt-Tiny model.
    Depths: [3, 3, 9, 3], Dimensions: [96, 192, 384, 768]
    """
    model = ConvNeXt(num_classes=num_classes, **kwargs)
    if pretrained:
        # Code to load weights can go here if needed
        url = "https://fbaipublicfiles.com"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == "__main__":
    # Example usage and sanity verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the tiny architecture
    model = convnext_tiny(depths=[2, 6, 2], dims=[36, 96, 164], num_classes=100).to(device)
    
    # Create dummy ImageNet batch: (batch_size=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model successfully initialized on: {device}")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
