import torch
import torch.nn as nn
from SIREN import Siren
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EdgeThreshold = 8.0/255.0
pi = 3.1415926
EPS = 1e-5

def generate_shader_const_array(lut_array, use_fp16=True):
    """生成shader常量数组"""

    dtype_str = "mediump float" if use_fp16 else "float"
    precision = "4" if use_fp16 else "6"

    with open("lgdsr_weight.fragment", "w") as f:
        f.write(f"""// 超低延迟MLP查找表 - 编译时常量
// 内存占用: {lut_array.nbytes} bytes
// 数据类型: {"FP16" if use_fp16 else "FP32"}

#version 450

const {dtype_str} MLP_LUT[32][15] = {{
""")

        for i in range(32):
            f.write(f"    {{")

            # 写入15个值
            for j in range(15):
                value = lut_array[i, j]
                if j < 14:
                    f.write(f" {value:.{precision}f},")
                else:
                    f.write(f" {value:.{precision}f}")

            f.write(" }")
            if i < 31:
                f.write(",")
            f.write("\n")

        f.write("};\n\n")

class LGDSR_rgb(nn.Module):
    def __init__(self, device, scale_factor=1.5, sample=5, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.device = device
        self.sample = sample
        self.linear = Siren(in_features=2, hidden_features=[128, 128, 64], hidden_layers=2, out_features=3 * sample,
                            outermost_linear=True)
        self.delta_init = torch.tensor([[0., -1.], [-1., 0.], [0., 0.], [1., 0.], [0., 1.]], device=device).unsqueeze(0).unsqueeze(1).unsqueeze(2) # [1, 1, 1, 5, 2]
        self.weight_init = torch.tensor([0.15, 0.15, 0.4, 0.15, 0.15], device=device).unsqueeze(0).unsqueeze(1).unsqueeze(2) # [1, 1, 1, 5]
        self.EPS = EPS
        self.pi = pi
        self.edgeThreshold = EdgeThreshold
        self.align = align_corners

    def fix_map(self, use_fp16=True):
        all_results = []
        for slope_idx in range(8):
            for strength_idx in range(4):
                # 量化输入
                slope = slope_idx / 8.0
                strength_values = [0.06, 0.12, 0.24, 0.48]
                strength = strength_values[strength_idx]

                # MLP推理
                mlp_input = torch.tensor([slope, strength]).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
                mlp_output = self.linear(mlp_input)

                # 解析输出
                pos_off_raw = mlp_output[..., :10].reshape(5, 2)
                weight_raw = mlp_output[..., 10:]

                # 添加初始值
                pos_off = pos_off_raw + self.delta_init
                weight = weight_raw + self.weight_init

                # 合并为单个向量 [pos_off(10) + weight(5)] = 15
                combined = torch.cat([pos_off.flatten(), weight.flatten()])
                all_results.append(combined.detach().cpu().numpy())

        # 转换为numpy数组
        lut_array = np.array(all_results)  # [32, 15]

        # 生成shader代码
        generate_shader_const_array(lut_array, use_fp16)

        return lut_array

if __name__=="__main__":
    model_path = r'D:\s00827220\projects\FSR_GSR_Test\ckpts\model_mlp.pth'
    state_dict = torch.load(model_path)['model']
    lgdsr = LGDSR_rgb(device, scale_factor=1.5, align_corners=False)
    lgdsr.load_state_dict(state_dict)
    lgdsr = lgdsr.to(device)
    lgdsr.eval()
    lgdsr.fix_map()