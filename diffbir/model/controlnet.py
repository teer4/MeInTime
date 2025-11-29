import torch
import torch as th
import torch.nn as nn

from .util import conv_nd, linear, zero_module, timestep_embedding, exists
from .attention import SpatialTransformer
from .unet import (
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
    UNetModel,
)
from .gated_fusion import GatedResidualFusion



class ControlledUnetModel(UNetModel):

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        image_prompt=None,
        lambda_t=None,
        control=None,
        only_mid_control=False,
        **kwargs,
    ):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        #print(t_emb.shape)  #torch.Size([1, 320])
        emb = self.time_embed(t_emb)
        #print(emb.shape)    #torch.Size([1, 1280])
        h, emb, context, image_prompt, lambda_t = map(lambda t: t.type(self.dtype), (x, emb, context, image_prompt, lambda_t))
        for module in self.input_blocks:
            h = module(h, emb, context, image_prompt, lambda_t)
            hs.append(h)
        h = self.middle_block(h, emb, context, image_prompt, lambda_t)

        #middle_block输出与control最后一个门控融合
        if control is not None:
            h = self.control_fusions[0](h, control.pop())
      
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                #门控融合
                h_skip = self.control_fusions[i + 1](hs.pop(), control.pop())  # 注意 +1 对齐注册索引
                h = torch.cat((h, h_skip), dim=1)
            h = module(h, emb, context, image_prompt, lambda_t)

        h = h.type(x.dtype)
        #输出controlnet控制的unet输出
        return self.out(h)
"""
[UpBlock 0] h.shape: torch.Size([1, 1280, 8, 8]), skip.shape: torch.Size([1, 1280, 8, 8]), control.shape: torch.Size([1, 1280, 8, 8])
[UpBlock 1] h.shape: torch.Size([1, 1280, 8, 8]), skip.shape: torch.Size([1, 1280, 8, 8]), control.shape: torch.Size([1, 1280, 8, 8])
[UpBlock 2] h.shape: torch.Size([1, 1280, 8, 8]), skip.shape: torch.Size([1, 1280, 8, 8]), control.shape: torch.Size([1, 1280, 8, 8])
[UpBlock 3] h.shape: torch.Size([1, 1280, 16, 16]), skip.shape: torch.Size([1, 1280, 16, 16]), control.shape: torch.Size([1, 1280, 16, 16])
[UpBlock 4] h.shape: torch.Size([1, 1280, 16, 16]), skip.shape: torch.Size([1, 1280, 16, 16]), control.shape: torch.Size([1, 1280, 16, 16])
[UpBlock 5] h.shape: torch.Size([1, 1280, 16, 16]), skip.shape: torch.Size([1, 640, 16, 16]), control.shape: torch.Size([1, 640, 16, 16])
[UpBlock 6] h.shape: torch.Size([1, 1280, 32, 32]), skip.shape: torch.Size([1, 640, 32, 32]), control.shape: torch.Size([1, 640, 32, 32])
[UpBlock 7] h.shape: torch.Size([1, 640, 32, 32]), skip.shape: torch.Size([1, 640, 32, 32]), control.shape: torch.Size([1, 640, 32, 32])
[UpBlock 8] h.shape: torch.Size([1, 640, 32, 32]), skip.shape: torch.Size([1, 320, 32, 32]), control.shape: torch.Size([1, 320, 32, 32])
[UpBlock 9] h.shape: torch.Size([1, 640, 64, 64]), skip.shape: torch.Size([1, 320, 64, 64]), control.shape: torch.Size([1, 320, 64, 64])
[UpBlock 10] h.shape: torch.Size([1, 320, 64, 64]), skip.shape: torch.Size([1, 320, 64, 64]), control.shape: torch.Size([1, 320, 64, 64])
[UpBlock 11] h.shape: torch.Size([1, 320, 64, 64]), skip.shape: torch.Size([1, 320, 64, 64]), control.shape: torch.Size([1, 320, 64, 64])
"""

class ControlNet(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_ip_adapter=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            #将num_res_blocks扩展成与channel_mult([1, 2, 4, 4])相同长度的列表，[2, 2, 2, 2]
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_ip_adapter = use_ip_adapter
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        dims, in_channels + hint_channels, model_channels, 3, padding=1
                    )
                )
            ]
        )

        #刚开始输入控制图的零卷积
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)]) #320

        self._feature_size = model_channels     #记录累计的特征总维度值，doesn't matter
        input_block_chans = [model_channels]    #记录每一层通道数
        ch = model_channels                     #当前通道数
        ds = 1                                  #当前下采样倍数

        #遍历不同通道层[1, 2, 4, 4] -> [320, 640, 1280, 1280], level是索引0， 1， 2， 3
        #一次循环结束（一层通道）：input_blocks: TimestepEmbedSequential{ (ResNetBlock + Attentionblock/SpatialTransformer) * 2 + Downsample }
        for level, mult in enumerate(channel_mult):

            #遍历当前通道层每个大block(ResNetBlock + Attentionblock), num_res_blocks[0] = 2
            #构建(ResNetBlock + Attentionblock/SpatialTransformer)*2
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                use_ip_adapter = use_ip_adapter,
                            )
                        )
                #input_blocks: TimestepEmbedSequential(ResNetBlock + Attentionblock/SpatialTransformer)
                self.input_blocks.append(TimestepEmbedSequential(*layers))

                #每一大块(ResNetBlock + Attentionblock)后做一次零卷积
                self.zero_convs.append(self.make_zero_conv(ch)) #[320, 320, 640, 640, 1280, 1280, 1280, 1280]
                self._feature_size += ch
                #记录当前通道数到input_block_chans，供skip connection使用
                input_block_chans.append(ch)
            
            #不是最后一层（4）执行下采样
            if level != len(channel_mult) - 1:
                #下采样不改变通道数，out_ch = ch
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                #下采样后做一次零卷积
                self.zero_convs.append(self.make_zero_conv(ch)) #[320, 640, 1280]
                ds *= 2    #更新下采样倍率
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        
        #middle_block: TimestepEmbedSequential (ResNetBlock + Attentionblock/SpatialTransformer + ResNetBlock)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                    use_ip_adapter = use_ip_adapter,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)  #1280
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        #DiffBIR对controlnet输入做x_nosiy和条件图的concat操作
        x = torch.cat((x, hint), dim=1)
        outs = []

        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        #input_blocks: TimestepEmbedSequential(ResNetBlock + Attentionblock/SpatialTransformer) * 2 + TimestepEmbedSequential(Downsample)
        #每个module是一个TimestepEmbedSequential，对应一个zero_conv
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        #输出残差列表
        return outs
