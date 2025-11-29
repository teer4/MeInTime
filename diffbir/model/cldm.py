from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook
from diffbir.ip_adapter.resampler import Resampler


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):
    
    #由self.cldm: ControlLDM = instantiate_from_config触发自动执行
    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor, image_proj_model_cfg
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13
        self.image_proj_model = Resampler(**image_proj_model_cfg)

    #将 checkpoint 文件中的权重参数注入到实例化的模型模块（unet, vae, clip）中
    @torch.no_grad()
    def load_pretrained_sd(self, sd: Dict[str, torch.Tensor]) -> Tuple[Set[str], Set[str]]:   #sd是checkpoint权重dict 
        #模块与checkpoint路径映射表，key是自定义self.unet，value是checkpoint权重参数前缀
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        #自定义模块清单，config实例化的模型对象，还没加载权重
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        #开始匹配并加载权重，name：“unet”，module：实例化对象self.unet
        for name, module in modules:
            init_sd = {}
            #scratch_sd：模块初始化后的空参数字典
            scratch_sd = module.state_dict()
            """
            with open("custom_unet_state_dict.txt", "w") as f:
                for name, param in scratch_sd.items():
                    f.write(f"{name}: {tuple(param.shape)}\n")
            """       
            #遍历模块每个参数名
            #module_map["unet"] = "model.diffusion_model", key = "input_block.0.0.weight", target_key = model.diffusion_model.input_block.0.0.weight
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                #target_key去匹配checkpoint
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                #从checkpoint复制匹配的模块参数权重
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            #将参数权重字典注入当前模型结构
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        #vae, unet, clip预训练模型需要冻结，关闭梯度，进入eval模式
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    #从unet提取兼容参数初始化controlnet
    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        #还没有预训练权重，加载了网络结构（layers），每一层的参数（weight、bias）使用 PyTorch 默认规则初始化
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}                  #最后要用于加载到 ControlNet 的权重
        init_with_new_zero = set()    #记录哪些权重是【零补齐】的
        init_with_scratch = set()     #记录哪些权重是【直接用 ControlNet 默认值】的
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                #如果两者形状完全一样，直接克隆 UNet 的权重
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    #计算需要补多少通道
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    #创建补零的张量
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    #把原UNet权重（target）和零张量（zeros）在输入通道（dim=1）方向上拼接起来
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        #加载新的权重到ControlNet
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    @torch.no_grad()
    def load_adapter_from_ckpt(self, adapter_ckpt: Dict[str, torch.Tensor]):
        # 加载 image_proj_model
        self.image_proj_model.load_state_dict(adapter_ckpt["image_proj_model"], strict=True)
        # 加载 UNet 的 to_k_ip / to_v_ip
        for name, module in self.unet.named_modules():
            if name in adapter_ckpt["ip_attn"]:
                module.load_state_dict(adapter_ckpt["ip_attn"][name], strict=True)
        # 加载 UNet 的 control_fusions (含门控)
        if "control_fusions" in adapter_ckpt:
            for name, module in self.unet.named_modules():
                if name in adapter_ckpt["control_fusions"]:
                    module.load_state_dict(adapter_ckpt["control_fusions"][name], strict=True)

    #输入gt是512, RGB, [-1, 1], float32, [B, C, H, W]
    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,                                                                                                                                      
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            #encoder(image)返回值是一个DiagonalGaussianDistribution对象，里面包含当前图像后验分布均值，方差，标准差的计算结果
            #sample()重参数化技巧采样，引入误差
            z = encoder(image).sample() * self.scale_factor
        else:
            #mode()取均值作为结果
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(
                cond_img * 2 - 1,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),
        )


    def encode_image_prompt(self, face_embedding, device: torch.device = "cuda") -> torch.Tensor:
        '''
        将 InsightFace 提取的人脸特征向量转换为可注入 UNet CrossAttention 的 image_prompt
        
        Args:
            face_embedding : shape [B, 512]
            device (torch.device): 模型与输入所在的设备（默认: "cuda"）

        Returns:
            torch.Tensor: 处理后的 image_prompt, shape [B, 16, 1024]
        '''
        if isinstance(face_embedding, torch.Tensor):
            face_embedding = face_embedding.clone().detach()
        else:
            face_embedding = torch.tensor(face_embedding)

        #print(face_embedding)  torch.Size([B, 512])
        #reshape成[B, length, image_dim] , [B, 1, 512]
        if face_embedding.ndim == 2:
            face_embedding = face_embedding.unsqueeze(1)  # [B, 512] → [B, 1, 512]
        elif face_embedding.ndim != 3:
            raise ValueError(f"Expected face_embedding to be [B, 512] or [B, 1, 512], got {face_embedding.shape}")
        
        if not hasattr(self, "image_proj_model"):
            raise ValueError("Image projection model (Resampler) not set in CLDM.")

        return self.image_proj_model(face_embedding)



    def forward(self, x_noisy, t, cond, image_prompt, lambda_t):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        #print(c_txt.shape)   #torch.Size([1, 77, 1024])
        #print(c_img.shape)   #torch.Size([1, 4, 64, 64])
        #print(t.shape)       #torch.Size([1]) 
        control = self.controlnet(x=x_noisy, hint=c_img, timesteps=t, context=c_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            image_prompt=image_prompt,
            control=control,
            only_mid_control=False,
            lambda_t=lambda_t,
        )
        return eps

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
