import torch
import torch.nn.functional as F
from fsrBase import FSR

EPS = 1e-5
FSR_RCAS_LIMIT = 0.25 - 1.0 / 16.0

class FSR_S(FSR):
    def __init__(self, device, scale_factor=2.0):
        self.scale_factor = scale_factor
        self.device = device
        self.pus = nn.PixelUnshuffle(2)
        self.ps = nn.PixelShuffle(2)

    def pre_input(self, lr):
        lumen = self.transfergray(lr)
        lumen_pus = self.pus(lumen)
        return lumen_pus

    def easu_process(self, lr, kernel):
        B, C, H, W = lr.shape
        dst_H = int(H * self.scale_factor)
        dst_W = int(W * self.scale_factor)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(dst_H, dtype=lr.dtype, device=self.device),
            torch.arange(dst_W, dtype=lr.dtype, device=self.device),
            indexing='ij'
        )
        srcx_f = (x_coords + 0.5) / self.scale_factor - 0.5
        srcy_f = (y_coords + 0.5) / self.scale_factor - 0.5
        src_x = torch.floor(srcx_f)
        src_y = torch.floor(srcy_f)

        # lumen = self.transfergray(lr)
        coff = torch.stack(self.get_near_block(lr, src_x, src_y), dim=0)
        # fl, gl, jl, kl, bl, cl, el, hl, il, ll, nl, ol = self.get_near_block(lumen, src_x, src_y)

        u = srcx_f - src_x
        v = srcy_f - src_y

        dir_x = kernel[:, 0:1]
        dir_y = kernel[:, 1:2]
        length = kernel[:, 2:3]
        # Normalize direction
        dirr = dir_x * dir_x + dir_y * dir_y
        mask = dirr < 1.0 / 32768.0
        dirr = torch.where(mask, torch.ones_like(dirr), dirr)
        dir_x = torch.where(mask, torch.ones_like(dir_x), dir_x)
        dirr = 1 / torch.sqrt(dirr)
        dir_x = dir_x * dirr
        dir_y = dir_y * dirr
        length = (length * 0.5) ** 2

        stretch = (dir_x * dir_x + dir_y * dir_y) / torch.maximum(torch.abs(dir_x), torch.abs(dir_y))
        ## 将 F 值范围从 (0, 2) 转换到 (0, 1)

        len2 = (1 + (stretch - 1) * length, 1 - 0.5 * length)
        # w = 1/2 - 1/4 * F   w ∈ [1/4, 1/2]
        lob = 0.5 + ((1.0 / 4 - 0.04) - 0.5) * length
        # clob  1/w    clob ∈ [1, 2]
        clob = 1.0 / lob

        rgb_min, rgb_max = torch.aminmax(coff[:4], dim=0)

        w = self.cal_color_weight(u, v, dir_x, dir_y, len2, lob, clob)
        wa = torch.sum(w, dim=0)
        ca = torch.sum(w * coff, dim=0)
        wa = torch.clamp(wa, min=EPS)
        result = torch.clamp(ca/wa, min=rgb_min, max=rgb_max)
        return result

    def rcas_process(self, img, sharpness_kernel, denoise=True):
        e, b, d, f, h = self.get_near_block_five(img)
        max_rgb = torch.maximum(torch.maximum(torch.maximum(b, d), f), h)
        min_rgb = torch.minimum(torch.minimum(torch.minimum(b, d), f), h)
        hitmin = torch.minimum(min_rgb, e) / (4 * max_rgb + EPS)
        hitmax = (1.0 - torch.maximum(max_rgb, e)) / (4.0 * min_rgb - 4.0 + EPS)
        lobe_rgb = torch.maximum(-hitmin, hitmax)
        lobe = torch.amax(lobe_rgb, dim=1, keepdim=True)
        lobe = torch.clip(lobe, -FSR_RCAS_LIMIT, 0.0)
        lobe *= sharpness_kernel  # 应用锐化强度
        # if denoise:
        #     lobe *= nz
        # 计算加权平均
        denom = 4.0 * lobe + 1.0
        rcpL = torch.where(torch.abs(denom) < 1e-5, torch.ones_like(denom), 1.0 / denom)
        output = (lobe * (b + d + f + h) + e) * rcpL
        return output
