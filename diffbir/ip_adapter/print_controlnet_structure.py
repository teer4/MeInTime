import torch
from diffusers import ControlNetModel

def print_module_tree(module, prefix=""):
    for name, child in module.named_children():
        print(f"{prefix}- {name}: {child.__class__.__name__}")
        print_module_tree(child, prefix + "  ")

if __name__ == "__main__":
    print(">> Loading ControlNetModel from v1_face.pth...")

    # 先初始化结构
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    # 加载你自己的权重
    state_dict = torch.load("/DiffBIR/v1_face.pth", map_location="cpu")
    controlnet.load_state_dict(state_dict)

    # 打印结构
    print("\n====== [CONTROLNET STRUCTURE] ======")
    print_module_tree(controlnet)
