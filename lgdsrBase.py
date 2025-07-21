import torch
import torch.nn as nn
import torch.nn.functional as F
from SIREN import Siren
import cv2
import numpy as np
import os
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EdgeThreshold = 8.0/255.0
pi = 3.1415926
EPS = 1e-5

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

    def cal_edgeDirection(self, f, g, j, k):
        # coff: fgjk
        delta_x = (k + g) - (f + j)
        delta_y = (k + j) - (f + g)

        slope = delta_y / (delta_x + self.EPS) # [N, 1, H, w]
        # slope_index: 0-3 strength_class: 0-1
        slope = slope.atan() / self.pi + 0.5
        strength = torch.sqrt(delta_x * delta_x + delta_y * delta_y) # [N, 1, H, W]

        return slope, strength

    def bilinear_upsample(self, img, grid, WH, align_corners=False):
        N, _, _, _ = img.shape
        if align_corners:
            grid = grid / (WH - 1) * 2 -1
        else:
            grid = (grid + 0.5) / WH * 2 - 1

        grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H_out, W_out, 2)
        return F.grid_sample(
            img,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners
        )

    def bilinear_grid_sample(self, img, grid, WH, delta_offset, align_corners=False):
        N, _, _, _ = img.shape
        # 归一化基础网格 (与原始方法一致)
        if align_corners:
            grid_normalized = grid / (WH - 1) * 2 -1
            scale = 2.0 / (WH - 1)
        else:
            grid_normalized = (grid + 0.5) / WH * 2 - 1
            scale = 2.0 / WH

        # 扩展基础网格到批次大小 (N, H_out, W_out, 2)
        grid_base = grid_normalized.unsqueeze(0).expand(N, -1, -1, -1)
        # 调整scale形状以匹配delta_offset (1,1,1,1,2)
        scale = scale.view(1, 1, 1, 1, 2)
        # 将像素偏移转换为归一化空间偏移 (N,5,H_out,W_out,2)
        delta_offset_norm = delta_offset * scale
        # 扩展基础网格并添加偏移 (N,5,H_out,W_out,2)
        final_grids = grid_base.unsqueeze(3) + delta_offset_norm
        # 合并偏移维度和高度维度 (N, 5*H_out, W_out, 2)
        outputs = []
        for i in range(self.sample):
            # 对第i个偏移位置进行采样
            sampled = F.grid_sample(
                img,
                final_grids[:, :, :, i],  # [N, H_out, W_out, 2]
                mode='bilinear',
                padding_mode='border',
                align_corners=align_corners
            )  # [N, C, H_out, W_out]
            outputs.append(sampled)
        # 堆叠所有采样结果
        stacked_outputs = torch.stack(outputs, dim=-1)  # [N, 5, C, H_out, W_out]
        return stacked_outputs

    ## f是我们需要采样的原图中的点
    ## f g
    ## j k
    def get_near_block(self, src_img, src_x, src_y, channel=1):
        B, C, H, W = src_img.shape
        # Generate normalized grid coordinates for all 12 points
        grid_points = [
            # f, g, j, k (center)
            (0, 0), (1, 0), (0, 1), (1, 1)
        ]

        sampled_points = []
        for dx, dy in grid_points:
            # Clamp coordinates to valid range
            x_clamped = torch.clamp(src_x + dx, 0, W - 1).long()
            y_clamped = torch.clamp(src_y + dy, 0, H - 1).long()

            output = src_img[:, channel:channel+1, y_clamped, x_clamped]
            sampled_points.append(output)

        return sampled_points

    def forward(self, lr):
        B, C, H, W = lr.shape
        dst_H = int(H * self.scale_factor)
        dst_W = int(W * self.scale_factor)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(dst_H, dtype=lr.dtype, device=self.device),
            torch.arange(dst_W, dtype=lr.dtype, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((x_coords, y_coords), dim=-1)
        WH = torch.tensor([dst_W, dst_H], dtype=lr.dtype, device=lr.device)

        pix = self.bilinear_upsample(lr, grid, WH, align_corners=self.align)

        srcx_f = (x_coords + 0.5) / self.scale_factor - 0.5
        srcy_f = (y_coords + 0.5) / self.scale_factor - 0.5
        src_x = torch.floor(srcx_f)
        src_y = torch.floor(srcy_f)
        fg, gg, jg, kg = self.get_near_block(lr, src_x, src_y)

        edgeVote = torch.abs(fg - jg) + torch.abs(pix[:, 1:2] - jg) + torch.abs(pix[:, 1:2] - fg)
        mask = edgeVote > self.edgeThreshold

        if mask.any():
            slope, strength = self.cal_edgeDirection(fg, gg, jg, kg)
            mlp_input = torch.cat([slope, strength], dim=1)
            mlp_output = self.linear(mlp_input.permute(0, 2, 3, 1))

            delta_init = self.delta_init.expand(B, dst_H, dst_W, -1, -1)
            pos_off = mlp_output[..., :10].reshape(B, dst_H, dst_W, self.sample, 2) + delta_init
            weight_init = self.weight_init.expand(B, dst_H, dst_W, -1)
            weight = mlp_output[..., 10:].reshape(B, dst_H, dst_W, self.sample) + weight_init
            weight_sum = weight.sum(dim=-1).unsqueeze(1)

            out_stack = self.bilinear_grid_sample(lr, grid, WH, pos_off, align_corners=self.align) * weight.unsqueeze(1)
            enhance = torch.sum(out_stack, dim=-1) / weight_sum

            res = torch.where(mask, enhance, pix)
        else:
            res = pix

        return res

    def forward_quantz(self, lr):
        B, C, H, W = lr.shape
        dst_H = int(H * self.scale_factor)
        dst_W = int(W * self.scale_factor)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(dst_H, dtype=lr.dtype, device=self.device),
            torch.arange(dst_W, dtype=lr.dtype, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((x_coords, y_coords), dim=-1)
        WH = torch.tensor([dst_W, dst_H], dtype=lr.dtype, device=lr.device)

        pix = self.bilinear_upsample(lr, grid, WH, align_corners=self.align)

        srcx_f = (x_coords + 0.5) / self.scale_factor - 0.5
        srcy_f = (y_coords + 0.5) / self.scale_factor - 0.5
        src_x = torch.floor(srcx_f)
        src_y = torch.floor(srcy_f)
        fg, gg, jg, kg = self.get_near_block(lr, src_x, src_y)

        edgeVote = torch.abs(fg - jg) + torch.abs(pix[:, 1:2] - jg) + torch.abs(pix[:, 1:2] - fg)
        mask = edgeVote > self.edgeThreshold

        if mask.any():
            slope, strength = self.cal_edgeDirection(fg, gg, jg, kg)

            # ===== STE梯度维持 ===== #
            # 斜率离散化
            slope_quantz_forward = torch.floor(slope * 8) / 8
            slope_quantz = slope_quantz_forward + (slope - slope.detach())

            # 强度离散化
            with torch.no_grad():
                strength_quantz_forward = torch.full_like(strength, 0.48)
                strength_quantz_forward[strength < 0.4] = 0.24
                strength_quantz_forward[strength < 0.18] = 0.12
                strength_quantz_forward[strength < 0.09] = 0.06

            strength_quantz = strength_quantz_forward + (strength - strength.detach())
            # ====================== #

            mlp_input = torch.cat([slope_quantz, strength_quantz], dim=1)
            mlp_output = self.linear(mlp_input.permute(0, 2, 3, 1))

            delta_init = self.delta_init.expand(B, dst_H, dst_W, -1, -1)
            pos_off = mlp_output[..., :10].reshape(B, dst_H, dst_W, self.sample, 2) + delta_init
            weight_init = self.weight_init.expand(B, dst_H, dst_W, -1)
            weight = mlp_output[..., 10:].reshape(B, dst_H, dst_W, self.sample) + weight_init
            weight_sum = weight.sum(dim=-1).unsqueeze(1)

            out_stack = self.bilinear_grid_sample(lr, grid, WH, pos_off, align_corners=self.align) * weight.unsqueeze(1)
            enhance = torch.sum(out_stack, dim=-1) / weight_sum

            res = torch.where(mask, enhance, pix)
        else:
            res = pix

        return res

# class LGDSR_resG:
#     def __init__(self, device, scale_factor=1.5):
#         self.scale_factor = scale_factor
#         self.device = device
#         self.edgeThreshold = EdgeThreshold
#
#     def fastLanczos2(self, x):
#         wa = x - 4
#         wb = x * wa - wa
#         wa = wa * wa
#         return wb * wa
#
#     def cal_edgeDirection(self, coff_mean):
#         # coff: fgjk  k-f  g-j
#         kf = coff_mean[:, 3:4] - coff_mean[:, 0:1]
#         gj = coff_mean[:, 1:2] - coff_mean[:, 2:3]
#         delta_x = kf + gj
#         delta_y = kf - gj
#         length_inv = torch.rsqrt(delta_x ** 2 + delta_y ** 2 + 3.075740e-05)
#
#         dir_x = delta_x * length_inv
#         dir_y = delta_y * length_inv
#         return dir_x, dir_y
#
#     def cal_weight_Y(self, u, v, coff_mean):
#         ffx = torch.stack([0.0 - u, 1.0 - u, 0.0 - u, 1.0 - u,
#                            0.0 - u, 1.0 - u, -1.0 - u, 2.0 - u,
#                            -1.0 - u, 2.0 - u, 0.0 - u, 1.0 - u], dim=0).unsqueeze(0)
#         ffy = torch.stack([0.0 - v, 0.0 - v, 1.0 - v, 1.0 - v,
#                            -1.0 - v, -1.0 - v, 0.0 - v, 0.0 - v,
#                            1.0 - v, 1.0 - v, 2.0 - v, 2.0 - v], dim=0).unsqueeze(0)
#
#         sum = torch.sum(torch.abs(coff_mean), dim=1, keepdim=True)
#         base_distance = ffx * ffx + ffy * ffy
#         if self.useEdgeDirection:
#             sumMean = 1.014185e+01 / (sum + EPS)
#             std = sumMean * sumMean
#             dir_x, dir_y = self.cal_edgeDirection(coff_mean[:, :4])
#             edgeDis = ffx * dir_y + ffy * dir_x
#             edge_factor = torch.clamp((coff_mean * coff_mean) * std, 0.0, 1.0) * 0.7 - 1.0
#             distance = base_distance + edgeDis * edgeDis * edge_factor
#         else:
#             sumMean = 1.014185 / (sum + EPS)
#             std = sumMean * sumMean
#             distance = base_distance * 0.55 + torch.clamp(torch.abs(coff_mean) * std, 0.0, 1.0)
#
#         w = self.fastLanczos2(distance)
#         finalY = torch.sum(w * coff_mean, dim=1, keepdim=True) / torch.sum(w, dim=1, keepdim=True)
#         min4, max4 = torch.aminmax(coff_mean[:, :4], dim=1, keepdim=True)
#         finalY = torch.clamp(self.edgeSharpness * finalY, min4, max4)
#
#         return finalY
#
#     def bilinear_upsample(self, img, grid, grid_HW, align_corners=False):
#         N, C, H_in, W_in = img.shape
#         if align_corners:
#             # 角点对齐模式
#             grid = grid / (grid_HW - 1) * 2 -1
#         else:
#             # 像素中心对齐模式 (默认)
#             grid = (grid + 0.5) / grid_HW * 2 - 1
#
#         grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H_out, W_out, 2)
#         return F.grid_sample(
#             img,
#             grid,
#             mode='bilinear',
#             padding_mode='border',
#             align_corners=align_corners
#         )
#
#     ## 12-tap kernel. f是我们需要采样的原图中的点
#     ##    b c
#     ##  e f g h
#     ##  i j k l
#     ##    n o
#     def get_near_block(self, src_img, src_x, src_y, channel=1):
#         B, C, H, W = src_img.shape
#         # Generate normalized grid coordinates for all 12 points
#         grid_points = [
#             # f, g, j, k (center)
#             (0, 0), (1, 0), (0, 1), (1, 1),
#             # b, c (top row)
#             (0, -1), (1, -1),
#             # e, h (middle row)
#             (-1, 0), (2, 0),
#             # i, l (next row)
#             (-1, 1), (2, 1),
#             # n, o (bottom row)
#             (0, 2), (1, 2)
#         ]
#
#         sampled_points = []
#         for dx, dy in grid_points:
#             # Clamp coordinates to valid range
#             x_clamped = torch.clamp(src_x + dx, 0, W - 1).long()
#             y_clamped = torch.clamp(src_y + dy, 0, H - 1).long()
#
#             output = src_img[:, channel:channel+1, y_clamped, x_clamped]
#             sampled_points.append(output)
#
#         return sampled_points
#
#     def lgdsr_process(self, lr):
#         B, C, H, W = lr.shape
#         dst_H = int(H * self.scale_factor)
#         dst_W = int(W * self.scale_factor)
#         y_coords, x_coords = torch.meshgrid(
#             torch.arange(dst_H, dtype=lr.dtype, device=self.device),
#             torch.arange(dst_W, dtype=lr.dtype, device=self.device),
#             indexing='ij'
#         )
#         grid = torch.stack((x_coords, y_coords), dim=-1)  # (H_out, W_out, 2)
#         grid_HW = torch.tensor([dst_W, dst_H], dtype=lr.dtype, device=lr.device)
#         pix = self.bilinear_upsample(lr, grid, grid_HW, align_corners=False)
#
#         srcx_f = (x_coords + 0.5) / self.scale_factor - 0.5
#         srcy_f = (y_coords + 0.5) / self.scale_factor - 0.5
#         src_x = torch.floor(srcx_f)
#         src_y = torch.floor(srcy_f)
#         fg, gg, jg, kg, bg, cg, eg, hg, ig, lg, ng, og = self.get_near_block(lr, src_x, src_y)
#         # shader if edgeVote > edgeThreshold, cal diff, other return pix
#         edgeVote = torch.abs(fg - jg) + torch.abs(pix[:, 1:2] - jg) + torch.abs(pix[:, 1:2] - fg)
#
#         mean = (jg + fg + kg + gg) * 0.25
#         pix_mean = pix[:, 1:2] - mean
#         coff = torch.cat([fg, gg, jg, kg, bg, eg], dim=1)
#         coff_mean = coff - mean
#
#         u = srcx_f - src_x
#         v = srcy_f - src_y
#         finalY = self.cal_weight_Y(u, v, coff_mean)
#         deltaY = torch.clamp(finalY - pix_mean, -23.0/255.0, 23.0/255.0)
#         enhance = torch.clamp(pix + deltaY, 0.0, 1.0)
#
#         mask = edgeVote > self.edgeThreshold
#         res = torch.where(mask, enhance, pix)
#         return res

def predict_lgdsr(test_path, model):
    from tqdm import tqdm
    predict_name = "{base_name}_1x5.png"
    for img_path in test_path:
        img_path_list = glob(os.path.join(img_path, "*.png"))
        img_path_list.sort()
        predict_path = os.path.join(img_path, 'lgdsr1x5_quantz_v2')
        os.makedirs(predict_path, exist_ok=True)
        for input in tqdm(img_path_list):
            base_name = os.path.basename(input)[:-4]
            img = cv2.imread(input, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lr_img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.set_grad_enabled(False):
                # output = model.forward(lr_img_tensor)
                output = model.forward_quantz(lr_img_tensor)
                pred = (np.clip(output[0].detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(predict_path, predict_name.format(base_name=base_name)), pred[..., ::-1])

if __name__=="__main__":
    test_path = [
        # r'E:\s00827220\works\SR_lut\dataset\verify_data\pbo_data47\540p',
        r'D:\s00827220\projects\FSR_GSR_Test\test\40_png',
        # r'D:\s00827220\projects\FSR_GSR_Test\test\cat'
    ]
    model_path = r'D:\s00827220\projects\FSR_GSR_Test\ckpts\model_mlp.pth'
    state_dict = torch.load(model_path)['model']
    lgdsr = LGDSR_rgb(device, scale_factor=1.5, align_corners=False)
    lgdsr.load_state_dict(state_dict)
    lgdsr = lgdsr.to(device)
    lgdsr.eval()
    predict_lgdsr(test_path, lgdsr)
