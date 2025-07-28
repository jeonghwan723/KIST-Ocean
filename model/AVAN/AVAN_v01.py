import torch.nn as nn
import torch, math
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

class AVAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.depths = config['depths']
        self.num_stages = config['num_stages']
        self.drop_path_rate = config['drop_path_rate']
        self.embed_dims = config['embed_dims']
        self.mlp_ratios = config['mlp_ratios']
        self.drop_rate = config['drop_rate']
        self.inp_zdim = config['inp_zdim']
        self.tar_zdim = config['tar_zdim']

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        cur = 0

        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()

        for i in range(self.num_stages):

            enc_layers = nn.Sequential(
                *[Block(dim=self.embed_dims[i], mlp_ratio=self.mlp_ratios[i], 
                        drop=self.drop_rate, drop_path=dpr[cur + j])
                    for j in range(self.depths[i])]
                    )

            self.enc_layers.append(enc_layers)

            enc_norm = LayerNorm(self.embed_dims[i], eps=1e-6, data_format="channels_first")            
            setattr(self, f"enc_norm{i + 1}", enc_norm)

            if i < self.num_stages - 1:

                dw_layers = nn.Sequential(DownSampling(self.embed_dims[i], self.embed_dims[i+1]))
                up_layers = nn.Sequential(UpSampling(self.embed_dims[-(i+1)] * 2 if i != 0 else self.embed_dims[-(i+1)], 
                                                    self.embed_dims[-(i+2)]))

                dec_layers = nn.Sequential(
                    *[Block(
                    dim=self.embed_dims[-(i+2)] * 2, mlp_ratio=self.mlp_ratios[-(i+2)], 
                    drop=self.drop_rate, drop_path=dpr[cur + j])
                    for j in range(self.depths[-(i+2)])])

                self.dec_layers.append(dec_layers)

                dec_norm = LayerNorm(self.embed_dims[-(i+2)] * 2, eps=1e-6, data_format="channels_first")
                
                setattr(self, f"dw_layers{i + 1}", dw_layers)
                setattr(self, f"up_layers{i + 1}", up_layers)
                setattr(self, f"dec_norm{i + 1}", dec_norm)
            
            cur += self.depths[i]

        self.stem = nn.Conv2d(self.inp_zdim, self.embed_dims[0], kernel_size=1,padding='same')
        self.head = nn.Conv2d(self.embed_dims[0] * 2, self.tar_zdim, kernel_size=1, padding='same')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, PartialConv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):

        # Encoder
        short = {}
        for i in range(self.num_stages):

            enc_layers = self.enc_layers[i]
            enc_norm = getattr(self, f"enc_norm{i + 1}")

            if i < self.num_stages - 1:
                dw_layers = getattr(self, f"dw_layers{i + 1}")
            
            x = enc_layers(x)
            x = enc_norm(x)

            if i < self.num_stages - 1:
                short[i] = x.clone()
                x = dw_layers(x)

        # Decoder
        for i in range(self.num_stages-1):

            dec_layers = self.dec_layers[i]
            dec_norm = getattr(self, f"dec_norm{i + 1}")
            up_layers = getattr(self, f"up_layers{i + 1}")

            x = up_layers(x)

            x = torch.cat([x, short[self.num_stages-2-i]], dim=1)

            x = dec_layers(x)
            x = dec_norm(x)

        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.forward_features(x)
        x = self.head(x)
        return x

class Disciminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tar_zdim = config['tar_zdim']
        self.drop_rate = config['drop_rate']

        # activation & dropout
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()

        # level 1
        self.enc11 = nn.Conv2d(self.tar_zdim, self.tar_zdim, kernel_size=5, padding='same', padding_mode='circular', groups=self.tar_zdim)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # level 2
        self.enc21 = nn.Conv2d(self.tar_zdim, self.tar_zdim, kernel_size=3, padding='same', padding_mode='circular', groups=self.tar_zdim)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # level 3
        self.enc31 = nn.Conv2d(self.tar_zdim, self.tar_zdim, kernel_size=3, padding='same', padding_mode='circular', groups=self.tar_zdim)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # level 4
        self.enc41 = nn.Conv2d(self.tar_zdim, self.tar_zdim, kernel_size=3, padding='same', padding_mode='circular', groups=self.tar_zdim)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # level 5
        self.enc51 = nn.Conv2d(self.tar_zdim, self.tar_zdim, kernel_size=3, padding='same', padding_mode='circular', groups=self.tar_zdim)
        self.enc52 = nn.Conv2d(self.tar_zdim, self.tar_zdim, kernel_size=3, padding='same', padding_mode='circular', groups=self.tar_zdim)

    def forward(self, x):

        # level 1
        x = self.enc11(x)
        x = self.act1(x)
        x = self.pool1(x)

        # level 2
        x = self.enc21(x)
        x = self.act1(x)
        x = self.pool2(x)

        # level 3
        x = self.enc31(x)
        x = self.act1(x)
        x = self.pool3(x)

        # level 4
        x = self.enc41(x)
        x = self.act1(x)
        x = self.pool4(x)

        # level 5
        x = self.enc51(x)
        x = self.enc52(x)
        x = self.act2(x)

        return x

class LKA(nn.Module):
    def __init__(self, dim, k=5, d=2):
        super(LKA, self).__init__()
        k1 = (2 * d) - 1
        k2 = math.ceil(k / d)
        self.conv0 = PartialConv2d(dim, dim, k1, padding='same', padding_mode='circular', groups=dim)
        self.conv_spatial = PartialConv2d(dim, dim, k2, stride=1, padding='same', padding_mode='circular', groups=dim, dilation=d)
        # self.conv_spatial = nn.Conv2d(dim, dim, k2, stride=1, padding='same', padding_mode='circular', groups=dim, dilation=d)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_feature=None, act_layer=nn.GELU, drop=0.):
        super(FFN, self).__init__()
        # out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pwconv1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = PartialConv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.pwconv2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, PartialConv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.lka = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        short = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.lka(x)
        x = self.proj_2(x)
        x = x + short
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2), padding_mode='circular')
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

class DownSampling(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.emb = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding='same')

    def forward(self, x):

        x = self.pool(x)
        x = self.emb(x)
            
        return x

class UpSampling(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.convtr = nn.ConvTranspose2d(in_chan, in_chan, 2, stride=2)
        self.emb = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding='same')

    def forward(self, x):

        x = self.convtr(x)
        x = self.emb(x)
            
        return x

class LayerNorm(nn.Module):
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
            return nn.LayerNorm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                #==============
                ### original
                # self.update_mask = nn.functional.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, padding_mode=self.padding_mode, dilation=self.dilation, groups=1)
                #==============

                #==============
                ### modified
                # Calculate padding size for 'same' padding
                def calculate_same_padding(input_size, kernel_size, stride, dilation):
                    return ((input_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size) // 2

                # Assuming input tensor has shape (batch_size, channels, height, width)
                input_height, input_width = mask.shape[2], mask.shape[3]
                padding_height = calculate_same_padding(input_height, self.weight_maskUpdater.shape[2], self.stride[0], self.dilation[0])
                padding_width = calculate_same_padding(input_width, self.weight_maskUpdater.shape[3], self.stride[1], self.dilation[1])

                padded_mask = nn.functional.pad(mask, pad=(padding_width, padding_width, padding_height, padding_height), mode='circular')
                self.update_mask = nn.functional.conv2d(padded_mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=0, dilation=self.dilation, groups=1)
                #==============

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output

