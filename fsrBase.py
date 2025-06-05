import torch
import torch.nn.functional as F

EPS = 1e-5
FSR_RCAS_LIMIT = 0.25 - 1.0 / 16.0

class FSR(object):
    def __init__(self, device, scale_factor=2.0):
        self.scale_factor = scale_factor
        self.device = device

    def transfergray(self, img):
        # B,C,H,W // RGB
        return 0.5 * img[:, 0:1] + img[:, 1:2] + 0.5 * img[:, 2:3]

    ## 12-tap kernel. f是我们需要采样的原图中的点
    ##    b c
    ##  e f g h
    ##  i j k l
    ##    n o
    def get_near_block(self, src_img, src_x, src_y):
        B, C, H, W = src_img.shape
        # Generate normalized grid coordinates for all 12 points
        grid_points = [
            # f, g, j, k (center)
            (0, 0), (1, 0), (0, 1), (1, 1),
            # b, c (top row)
            (0, -1), (1, -1),
            # e, h (middle row)
            (-1, 0), (2, 0),
            # i, l (next row)
            (-1, 1), (2, 1),
            # n, o (bottom row)
            (0, 2), (1, 2)
        ]

        # Sample all points using direct indexing with clamping
        sampled_points = []

        for dx, dy in grid_points:
            # Clamp coordinates to valid range
            x_clamped = torch.clamp(src_x + dx, 0, W - 1).long()
            y_clamped = torch.clamp(src_y + dy, 0, H - 1).long()

            output = src_img[:, :, y_clamped, x_clamped]
            sampled_points.append(output)

        return sampled_points

    ##    b c         b        c
    ##  e f g h     e f g    f g h     f         g
    ##  i j k l       j        k     i j k     j k l
    ##    n o                          n         o
    def cal_dir_len(self, top, left, center, right, bot):
        x_right = torch.abs(right - center)
        x_left = torch.abs(center - left)
        dirx = right - left
        max_x= torch.clamp(torch.maximum(x_right, x_left), min=EPS)
        lenx = torch.clamp(torch.abs(dirx) / max_x, 0.0, 1.0)
        lenx = lenx * lenx

        y_bot = torch.abs(bot - center)
        y_top = torch.abs(center - top)
        diry = bot - top
        max_y = torch.clamp(torch.maximum(y_bot, y_top), min=EPS)
        leny = torch.clamp(torch.abs(diry) / max_y, 0.0, 1.0)
        leny = leny * leny
        length = lenx+leny

        rslt = torch.cat([dirx, diry, length], dim=1)
        return rslt

    def cal_color_weight(self, u, v, dir_x, dir_y, len2, lob, clob):
        # f, g, j, k, b, c, e, h, i, l, n, o
        ffx = torch.stack([0.0 - u, 1.0 - u, 0.0 - u, 1.0 - u,
                           0.0 - u, 1.0 - u, -1.0 - u, 2.0 - u,
                           -1.0 - u, 2.0 - u, 0.0 - u, 1.0 - u], dim=0).unsqueeze(1).unsqueeze(1)
        ffy = torch.stack([0.0 - v, 0.0 - v, 1.0 - v, 1.0 - v,
                           -1.0 - v, -1.0 - v, 0.0 - v, 0.0 - v,
                           1.0 - v, 1.0 - v, 2.0 - v, 2.0 - v], dim=0).unsqueeze(1).unsqueeze(1)
        # vx = x * cos + y * sin
        # vy = -x * sin + y * cos
        vx = ffx * dir_x + ffy * dir_y
        vy = ffx * (-dir_y) + ffy * dir_x

        vx = vx * len2[0]
        vy = vy * len2[1]

        d2 = vx * vx + vy * vy
        d2 = torch.clamp(d2, max=clob)

        ##  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
        ##  |        |____________|                 |   |               |
        ##  |              wb                       |   |      wa       |
        ##  |_______________________________________|   |_______________|
        ##                   base                             window

        wb = 2.0 / 5 * d2 -1
        wa = lob * d2 - 1
        wb = wb * wb
        wa = wa * wa
        base = 25.0 / 16 * wb - (25.0 / 16 - 1)
        w = base * wa
        return w

    def easu_process(self, lr):
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

        lumen = self.transfergray(lr)
        # f, g, j, k, b, c, e, h, i, l, n, o
        coff = torch.stack(self.get_near_block(lr, src_x, src_y), dim=0)
        fl, gl, jl, kl, bl, cl, el, hl, il, ll, nl, ol = self.get_near_block(lumen, src_x, src_y)
        u = srcx_f - src_x
        v = srcy_f - src_y

        # s (top-left)  t (top-right)  u (bottom-left)  v (bottom-right)
        w_s = (1 - u) * (1 - v)
        w_t = u * (1 - v)
        w_u = (1 - u) * v
        w_v = u * v
        dir_lens = torch.stack([self.cal_dir_len(bl, el, fl, gl, jl) * w_s,
                               self.cal_dir_len(cl, fl, gl, hl, kl) * w_t,
                               self.cal_dir_len(fl, il, jl, kl, nl) * w_u,
                               self.cal_dir_len(gl, jl, kl, ll, ol) * w_v], dim=-1)
        dir_len = torch.sum(dir_lens, dim=-1)
        dir_x = dir_len[:, 0:1]
        dir_y = dir_len[:, 1:2]
        length = dir_len[:, 2:3]
        # Normalize direction
        dirr = dir_x * dir_x + dir_y * dir_y
        mask = dirr < 1.0 / 32768.0
        dirr = torch.where(mask, torch.ones_like(dirr), dirr)
        dir_x = torch.where(mask, torch.ones_like(dir_x), dir_x)
        dirr = 1 / torch.sqrt(dirr)
        dir_x = dir_x * dirr
        dir_y = dir_y * dirr

        stretch = (dir_x * dir_x + dir_y * dir_y) / torch.maximum(torch.abs(dir_x), torch.abs(dir_y))
        ## 将 F 值范围从 (0, 2) 转换到 (0, 1)
        length = (length * 0.5) ** 2
        len2 = (1 + (stretch - 1) * length, 1 - 0.5 * length)
        # w = 1/2 - 1/4 * F   w ∈ [1/4, 1/2]
        lob = 0.5 + ((1.0 / 4 - 0.04) - 0.5) * length
        # clob  1/w    clob ∈ [1, 2]
        clob = 1.0 / lob

        rgb_min, rgb_max = torch.aminmax(coff[:4], dim=0)

        w = self.cal_color_weight(u, v, dir_x, dir_y, len2, lob, clob)
        wa = torch.sum(w, dim=0)
        ca = torch.sum(w*coff, dim=0)
        wa = torch.clamp(wa, min=EPS)
        result = torch.clamp(ca/wa, min=rgb_min, max=rgb_max)
        return result

    ## e是中心点, 输入为easu处理后的高分辨图，不需要网格
    ##    b
    ##  d e f
    ##    h
    def get_near_block_five(self, img):
        B, C, H, W = img.shape
        img_pad = F.pad(img, [1, 1, 1, 1], mode="replicate")
        # e, b, d, f, h
        offest = [(1, 1), (1, 0), (0, 1), (2, 1), (1, 2)]

        sampled_points = []
        for x_offset, y_offset in offest:
            # Initialize output tensor
            output = img_pad[:, :, y_offset:H+y_offset, x_offset:W+x_offset]
            sampled_points.append(output)

        return sampled_points

    def rcas_process(self, img, sharpness_para=0.0, denoise=True):
        sharpness = 2 ** (-sharpness_para)
        lumen = self.transfergray(img)
        e, b, d, f, h = self.get_near_block_five(img)
        el, bl, dl, fl, hl = self.get_near_block_five(lumen)
        # 噪声检测
        nz = 0.25 * (bl + dl + fl + hl) - el
        max_luma = torch.maximum(torch.maximum(torch.maximum(torch.maximum(bl, dl), el), fl), hl)
        min_luma = torch.minimum(torch.minimum(torch.minimum(torch.minimum(bl, dl), el), fl), hl)
        range_luma = torch.clamp(max_luma - min_luma, min=EPS)  # 避免除零
        nz = torch.abs(nz) / range_luma
        nz = torch.clamp(nz, 0.0, 1.0)
        nz = (-0.5) * nz + 1.0
        max_rgb = torch.maximum(torch.maximum(torch.maximum(b, d), f), h)
        min_rgb = torch.minimum(torch.minimum(torch.minimum(b, d), f), h)
        hitmin = torch.minimum(min_rgb, e) / (4 * max_rgb + EPS)
        hitmax = (1.0 - torch.maximum(max_rgb, e)) / (4.0 * min_rgb - 4.0 + EPS)
        lobe_rgb = torch.maximum(-hitmin, hitmax)
        lobe = torch.amax(lobe_rgb, dim=1, keepdim=True)
        lobe = torch.clip(lobe, -FSR_RCAS_LIMIT, 0.0)
        lobe *= sharpness  # 应用锐化强度
        if denoise:
            lobe *= nz
        # 计算加权平均
        denom = 4.0 * lobe + 1.0
        rcpL = torch.where(torch.abs(denom) < 1e-5, torch.ones_like(denom), 1.0 / denom)
        output = (lobe * (b + d + f + h) + e) * rcpL
        return output
