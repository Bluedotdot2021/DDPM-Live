
import torch
import torch.nn as nn


def get_time_embedding(t, time_emb_dim): # t:[b]
    t_emb = t[:,None].repeat(1, time_emb_dim // 2) # t_emb:[b, time_emb_dim // 2]
    factor = 1000 ** (torch.arange(0, time_emb_dim // 2, dtype=torch.float32, device=t.device) / (time_emb_dim // 2))
    t_emb = t_emb / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb #[b, time_emb_dim]

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, down_sample):
        super().__init__()
        self.num_layers = num_layers

        self.norm_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.norm_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                )
                for _ in range(num_layers)
            ]
        )

        self.time_emb = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
                for i in range(num_layers)
            ]
        )

        self.attn_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                #original data: (batch_size, channels, h*w), convert to: (batch_size, h*w, channels)
                nn.MultiheadAttention(out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            residual_input = out
            out = self.norm_conv_first[i](out)
            #out:(b,c,h,w), self.time_emb[i](t_emb):(b,c), convert to: (b, c, 1, 1)
            out = out + self.time_emb[i](t_emb)[:,:,None, None]
            out = self.norm_conv_second[i](out)
            out = out + self.residual_input_conv[i](residual_input)

            residual_input = out
            b, c, h, w = out.shape
            attn_in = out.reshape(b, c, h*w)
            attn_in = self.attn_norms[i](attn_in)
            attn_in = attn_in.transpose(1,2) #(b,c,h*w)->(b,h*w,c)
            attn_out,_ = self.attentions[i](attn_in, attn_in, attn_in)
            attn_out = attn_out.transpose(1,2) #(b,h*w,c)->(b,c,h*w)
            attn_out = attn_out.reshape(b,c,h,w)
            out = attn_out + residual_input

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.norm_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1)
                )
                for i in range(num_layers + 1)
            ]
        )

        self.norm_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                )
                for _ in range(num_layers+1)
            ]
        )

        self.time_emb = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
                for i in range(num_layers + 1)
            ]
        )

        self.attn_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, t_emb):
        out = x
        out = self.norm_conv_first[0](out)
        out = out + self.time_emb[0](t_emb)[:,:,None,None]
        out = self.norm_conv_second[0](out)
        out = out + self.residual_input_conv[0](x)

        for i in range(self.num_layers):
            b,c,h,w = out.shape
            attn_in = out.reshape(b, c, h*w)
            attn_in = self.attn_norms[i](attn_in)
            attn_in = attn_in.transpose(1,2) # (b, c, h*w) to (b, h*w, c)
            attn_out, _ = self.attentions[i](attn_in, attn_in, attn_in)
            attn_out = attn_out.transpose(1, 2) # (b, h*w, c) to (b, c, h*w)
            attn_out = attn_out.reshape(b, c, h, w)
            out = out + attn_out

            residual_input = out
            out = self.norm_conv_first[i+1](out)
            out = out + self.time_emb[i+1](t_emb)[:,:,None, None]
            out = self.norm_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](residual_input)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, up_sample):
        super().__init__()
        self.num_layers = num_layers

        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if up_sample else nn.Identity()

        self.time_emb = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1)
                )
                for i in range(num_layers)
            ]
        )

        self.norm_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1)
                )
                for _ in range(num_layers)
            ]
        )

        self.attn_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, down_out, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, down_out], dim=1) #(b, c, h, w), concat on dimension c

        out = x
        for i in range(self.num_layers):
            residual_input = out
            out = self.norm_conv_first[i](out)
            out = out + self.time_emb[i](t_emb)[:,:,None, None]
            out = out + self.residual_input_conv[i](residual_input)

            b, c, h, w = out.shape
            attn_in = out.reshape(b, c, h*w)
            attn_in = self.attn_norms[i](attn_in)
            attn_in = attn_in.transpose(1, 2) # conver to (b, h*w, c)
            attn_out, _ = self.attentions[i](attn_in, attn_in, attn_in)
            attn_out = attn_out.transpose(1, 2)# conver to (b, c, h*w)
            attn_out = attn_out.reshape(b, c, h, w)
            out = out + attn_out

        return out

class Unet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']

        self.down_layers = model_config['down_layers']
        self.mid_layers = model_config['mid_layers']
        self.up_layers = model_config['up_layers']

        self.down_sample = model_config['down_sample']
        self.time_emb_dim = model_config['time_emb_dim']
        self.num_heads = model_config['num_heads']
        self.img_channels = model_config['img_channels']

        self.in_conv = nn.Conv2d(self.img_channels, self.down_channels[0], 3, 1, 1)
        self.t_proj = nn.Sequential(
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.SiLU(),
                nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1): #[c1, c2, c3, c4]:[32, 64, 128, 256]
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.time_emb_dim,
                                        self.num_heads, self.down_layers, self.down_sample[i]))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1): #[c4,c4, c3]:[256, 256, 128]
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.time_emb_dim,
                                      self.num_heads, self.mid_layers))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i != 0 else 16, self.time_emb_dim,
                                    self.num_heads, self.up_layers, self.down_sample[i]))

        self.out_norm = nn.GroupNorm(8, 16)
        self.out_conv = nn.Conv2d(16, self.img_channels, 3, 1, 1)

    def forward(self, x, t): # x:[b, c, h, w], t:[b]
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.time_emb_dim) # embed t to t:[b, time_emb_dim]
        t_emb = self.t_proj(t_emb)
        out = self.in_conv(x) # convert [b, 1, h, w] to [b, c1, h, w]

        down_outs = []
        #out = x
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        #down_outs: [b, c1, h, w], [b, c2, h/2, w/2], [b, c3, h/4, w/4]
        #out: [b, c4, h/4, w/4]

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            out = up(out, down_outs.pop(), t_emb)

        out = self.out_norm(out)
        out = nn.SiLU()(out)
        out = self.out_conv(out)

        return out


