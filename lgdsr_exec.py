import torch
import torch.nn.functional as F
import cv2
import numpy as np
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EdgeThreshold = 8.0/255.0
pi = 3.1415926
EPS = 1e-5

MLP_LUT = [
    [ -0.0233, -1.1263, -1.2668, 0.0876, 0.0108, 0.0275, 1.1547, 0.0291, 0.0460, 1.1945, -0.3008, -0.3050, 2.2439, -0.3165, -0.3072 ],
    [ -0.1419, -1.3149, -1.3746, 0.0766, 0.0030, 0.0125, 1.2516, -0.0810, 0.0457, 1.3406, -0.3080, -0.2752, 2.2644, -0.2910, -0.3055 ],
    [ -0.1719, -1.4009, -1.5701, 0.2753, 0.0117, 0.0173, 1.2728, -0.3355, 0.0596, 1.3704, -0.3053, -0.1271, 2.0867, -0.1489, -0.3166 ],
    [ 0.1199, -0.8787, -0.5658, 1.0036, -0.0212, -0.0024, 1.3681, -0.3349, 1.2141, 0.7305, -0.3029, -0.3113, 1.2931, -0.0023, 0.0772 ],
    [ -0.1061, -1.1388, -1.2436, 0.2137, -0.0019, 0.0147, 1.2339, -0.1541, 0.1378, 1.2500, -0.2936, -0.3551, 2.3049, -0.3677, -0.2815 ],
    [ 0.1629, -0.9622, -1.1138, 0.3504, 0.0132, -0.0040, 1.2203, -0.2414, 0.1556, 1.2424, -0.3442, -0.3491, 2.1490, -0.2973, -0.2280 ],
    [ 0.4633, -0.4589, -0.2968, 0.4864, -0.0454, -0.0078, 1.0924, 1.0416, 0.8671, 0.7473, -0.4097, -0.4998, 1.1104, -0.1790, 0.3334 ],
    [ 0.4115, -0.6361, -0.3835, 0.6775, -0.0220, 0.0017, 1.3094, 0.3109, 1.0009, 0.7069, -0.3353, -0.5047, 1.0591, -0.1482, 0.3545 ],
    [ 0.1214, -0.9101, -1.0489, 0.2193, 0.0181, -0.0001, 1.0492, -0.0879, 0.0679, 1.1734, -0.3312, -0.3111, 2.0471, -0.3081, -0.2362 ],
    [ 0.3850, -0.4136, -0.3993, 0.4686, -0.0289, -0.0014, 1.0304, 0.7567, 0.7582, 0.7495, -0.4420, -0.5327, 1.3216, -0.1968, 0.2280 ],
    [ 0.3944, -0.3698, -0.3587, 0.3990, -0.0508, -0.0112, 0.9957, 0.9803, 0.7093, 0.7628, -0.4260, -0.5128, 1.1120, -0.1839, 0.3088 ],
    [ 0.3942, -0.3522, -0.3595, 0.4106, -0.0239, -0.0077, 1.1787, 0.5332, 0.8311, 0.6360, -0.4260, -0.5247, 1.1405, -0.1966, 0.3071 ],
    [ 0.1751, -1.2488, -1.1782, 0.0128, 0.0071, 0.0217, 1.2466, 0.1823, -0.1450, 1.2780, -0.3496, -0.2987, 2.2469, -0.2816, -0.3368 ],
    [ 0.2424, -1.1762, -1.0936, 0.1480, -0.0081, 0.0173, 1.2375, 0.1779, -0.2216, 1.2579, -0.3240, -0.3111, 2.0479, -0.2480, -0.2819 ],
    [ 0.5231, -0.3553, -0.4707, 0.4072, -0.0198, -0.0210, 1.0784, 1.1469, 0.7923, 0.8618, -0.4633, -0.4852, 1.2306, -0.2042, 0.3130 ],
    [ 0.6620, -0.2730, -0.6110, 0.3866, -0.0156, -0.0593, 1.1458, 1.1052, 0.7424, 0.9578, -0.4616, -0.4441, 1.1319, -0.1937, 0.3961 ],
    [ -0.0020, -1.2686, -1.2120, 0.0818, -0.0009, 0.0356, 1.1924, 0.1773, 0.0371, 1.2233, -0.2809, -0.3194, 2.2255, -0.3215, -0.2730 ],
    [ 0.0070, -1.3744, -1.3649, 0.1259, 0.0059, 0.0143, 1.3556, 0.0628, 0.0692, 1.4042, -0.2835, -0.3399, 2.3877, -0.3281, -0.2701 ],
    [ 0.0215, -0.6324, -1.3842, 0.0801, 0.0056, -0.0464, 1.4310, 0.1112, 0.9461, 1.1453, -0.2874, -0.2815, 1.9001, -0.2900, 0.0231 ],
    [ 0.0323, -0.4503, -1.1129, 0.1056, 0.0114, -0.0133, 1.3301, -0.1853, 1.1324, 0.8463, -0.3135, -0.2937, 1.7047, -0.2032, -0.0590 ],
    [ -0.2147, -1.1481, -1.2520, 0.1760, -0.0012, 0.0176, 1.2377, -0.0459, 0.2213, 1.2345, -0.3940, -0.2891, 2.3401, -0.2818, -0.3531 ],
    [ -0.2985, -1.0834, -1.2046, 0.0829, -0.0131, 0.0043, 1.2665, 0.0050, 0.3440, 1.1435, -0.3827, -0.2553, 2.0769, -0.2137, -0.3225 ],
    [ -0.4423, -0.5108, -1.0374, 0.0551, -0.0440, -0.0028, 0.7151, -0.6963, 0.6552, 0.4617, -0.4998, -0.0843, 1.3414, 0.2917, -0.5396 ],
    [ -0.4513, -0.4580, -0.9797, 0.0733, -0.0528, 0.0130, 0.7241, -0.7830, 0.6747, 0.3847, -0.5155, -0.1040, 1.3036, 0.3452, -0.5615 ],
    [ -0.2563, -0.8824, -1.0260, -0.0748, -0.0004, 0.0229, 0.9784, 0.1415, 0.1799, 0.9972, -0.3179, -0.2644, 1.9915, -0.2721, -0.3028 ],
    [ -0.2465, -0.7379, -0.8788, -0.2359, -0.0240, 0.0054, 0.9053, -0.0545, 0.4594, 0.6331, -0.3465, -0.2595, 1.7069, -0.0662, -0.4392 ],
    [ -0.2462, -0.5117, -0.8543, -0.1363, -0.0347, 0.0164, 0.6230, -0.5732, 0.4590, 0.3915, -0.5169, -0.1818, 1.3892, 0.2842, -0.5974 ],
    [ -0.3099, -0.5037, -0.9445, -0.0385, -0.0377, -0.0030, 0.7334, -0.7447, 0.4572, 0.4494, -0.5348, -0.1519, 1.3342, 0.2973, -0.5406 ],
    [ 0.1800, -1.2226, -1.2606, -0.2026, 0.0017, 0.0195, 1.2825, 0.1967, -0.0764, 1.2300, -0.2743, -0.3763, 2.3369, -0.3450, -0.2878 ],
    [ -0.0904, -1.0348, -1.1334, -0.2781, -0.0059, 0.0052, 1.1308, 0.2824, 0.0444, 1.1655, -0.3015, -0.3537, 2.1550, -0.3271, -0.2702 ],
    [ -0.3704, -0.6665, -1.1468, -0.0369, -0.0136, 0.0263, 0.7716, -0.6926, 0.4769, 0.5041, -0.5398, -0.0762, 1.4450, 0.2619, -0.5463 ],
    [ -0.4090, -0.7054, -1.1146, 0.3946, -0.0311, -0.0161, 0.9124, -0.7302, 0.5376, 0.6780, -0.6005, -0.0195, 1.2609, 0.3045, -0.4048 ]
]

class LGDSR_rgb_quantz():
    def __init__(self, device, scale_factor=1.5, sample=5, align_corners=False):
        self.scale_factor = scale_factor
        self.device = device
        self.sample = sample
        self.EPS = EPS
        self.pi = pi
        self.edgeThreshold = EdgeThreshold
        self.align = align_corners
        self.lookmap = torch.tensor(MLP_LUT).to(device)

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
            # 角点对齐模式
            grid = grid / (WH - 1) * 2 -1
        else:
            # 像素中心对齐模式 (默认)
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

    def lgdsr_process(self, lr):
        B, C, H, W = lr.shape
        dst_H = int(H * self.scale_factor)
        dst_W = int(W * self.scale_factor)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(dst_H, dtype=lr.dtype, device=self.device),
            torch.arange(dst_W, dtype=lr.dtype, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((x_coords, y_coords), dim=-1)  # (H_out, W_out, 2)
        WH = torch.tensor([dst_W, dst_H], dtype=lr.dtype, device=lr.device)
        pix = self.bilinear_upsample(lr, grid, WH, align_corners=self.align)

        srcx_f = (x_coords + 0.5) / self.scale_factor - 0.5
        srcy_f = (y_coords + 0.5) / self.scale_factor - 0.5
        src_x = torch.floor(srcx_f)
        src_y = torch.floor(srcy_f)
        fg, gg, jg, kg = self.get_near_block(lr, src_x, src_y)
        slope, strength = self.cal_edgeDirection(fg, gg, jg, kg)
        slope_quantz = torch.floor(slope * 8) / 8
        slope_quantz_index = torch.floor(slope * 8).long()

        mask1 = strength < 0.09
        mask2 = strength < 0.18
        mask3 = strength < 0.4
        strength_quantz_index = torch.tensor(3).expand(slope_quantz.shape).to(self.device)
        strength_quantz_index[mask3] = 2
        strength_quantz_index[mask2] = 1
        strength_quantz_index[mask1] = 0
        lut_idx = slope_quantz_index * 4 + strength_quantz_index
        map = self.lookmap[lut_idx]
        pos_off = map[..., :10].reshape(B, dst_H, dst_W, self.sample, 2)
        weight = map[..., 10:].reshape(B, dst_H, dst_W, self.sample)

        weight_sum = weight.sum(dim=-1).unsqueeze(1)
        # shader if edgeVote > edgeThreshold, cal diff, other return pix
        edgeVote = torch.abs(fg - jg) + torch.abs(pix[:, 1:2] - jg) + torch.abs(pix[:, 1:2] - fg)
        out_stack = self.bilinear_grid_sample(lr, grid, WH, pos_off, align_corners=self.align) * weight.unsqueeze(1)
        enhance = torch.sum(out_stack, dim=-1) / weight_sum

        mask = edgeVote > self.edgeThreshold
        res = torch.where(mask, enhance, pix)
        return res

def predict_lgdsr_quantz(test_path):
    from tqdm import tqdm
    lgdsr = LGDSR_rgb_quantz(device)
    predict_name = "{base_name}_1x5.png"
    for img_path in test_path:
        img_path_list = glob(os.path.join(img_path, "*.png"))
        img_path_list.sort()
        predict_path = os.path.join(img_path, 'lgdsr1x5_exec')
        os.makedirs(predict_path, exist_ok=True)
        for input in tqdm(img_path_list):
            base_name = os.path.basename(input)[:-4]
            img = cv2.imread(input, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lr_img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.set_grad_enabled(False):
                output = lgdsr.lgdsr_process(lr_img_tensor)
                pred = (np.clip(output[0].detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(predict_path, predict_name.format(base_name=base_name)), pred[..., ::-1])

if __name__=="__main__":
    import os
    device = torch.device("cuda")
    test_path = [
        # r'E:\s00827220\works\SR_lut\dataset\verify_data\pbo_data47\540p',
        r'D:\s00827220\projects\FSR_GSR_Test\test\40_png',
        # r'D:\s00827220\projects\FSR_GSR_Test\test\cat'
    ]
    predict_lgdsr_quantz(test_path)
