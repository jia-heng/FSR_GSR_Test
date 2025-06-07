import torch
import torch.nn.functional as F

EdgeThreshold = 8.0/255.0
EdgeSharpness = 2.0
OperationMode = 1
UseEdgeDirection = True
EPS = 1e-5

class SGSR:
    def __init__(self, device, scale_factor=2.0):
        self.scale_factor = scale_factor
        self.device = device
        self.edgeThreshold = EdgeThreshold
        self.edgeSharpness = EdgeSharpness
        self.useEdgeDirection = UseEdgeDirection

    def fastLanczos2(self, x):
        wa = x - 4
        wb = x * wa - wa
        wa = wa * wa
        return wb * wa

    def cal_edgeDirection(self, coff_mean):
        # coff:fgj  kk-f  g-j
        kf = coff_mean[:, 3:4] - coff_mean[:, 0:1]
        gj = coff_mean[:, 1:2] - coff_mean[:, 2:3]
        delta_x = kf + gj
        delta_y = kf - gj
        length_inv = torch.rsqrt(delta_x ** 2 + delta_y ** 2 + 3.075740e-05)

        dir_x = delta_x * length_inv
        dir_y = delta_y * length_inv
        return dir_x, dir_y

    def cal_weight_Y(self, u, v, coff_mean):
        ffx = torch.stack([0.0 - u, 1.0 - u, 0.0 - u, 1.0 - u,
                           0.0 - u, 1.0 - u, -1.0 - u, 2.0 - u,
                           -1.0 - u, 2.0 - u, 0.0 - u, 1.0 - u], dim=0).unsqueeze(0)
        ffy = torch.stack([0.0 - v, 0.0 - v, 1.0 - v, 1.0 - v,
                           -1.0 - v, -1.0 - v, 0.0 - v, 0.0 - v,
                           1.0 - v, 1.0 - v, 2.0 - v, 2.0 - v], dim=0).unsqueeze(0)

        sum = torch.sum(torch.abs(coff_mean), dim=1, keepdim=True)
        base_distance = ffx * ffx + ffy * ffy
        if self.useEdgeDirection:
            sumMean = 1.014185e+01 / (sum + EPS)
            std = sumMean * sumMean
            dir_x, dir_y = self.cal_edgeDirection(coff_mean[:, :4])
            edgeDis = ffx * dir_y + ffy * dir_x
            edge_factor = torch.clamp((coff_mean * coff_mean) * std, 0.0, 1.0) * 0.7 - 1.0
            distance = base_distance + edgeDis * edgeDis * edge_factor
        else:
            sumMean = 1.014185 / (sum + EPS)
            std = sumMean * sumMean
            distance = base_distance * 0.55 + torch.clamp(torch.abs(coff_mean) * std, 0.0, 1.0)

        w = self.fastLanczos2(distance)
        finalY = torch.sum(w * coff_mean, dim=1, keepdim=True) / torch.sum(w, dim=1, keepdim=True)
        min4, max4 = torch.aminmax(coff_mean[:, :4], dim=1, keepdim=True)
        finalY = torch.clamp(self.edgeSharpness * finalY, min4, max4)

        return finalY

    def bilinear_upsample(self, img, grid, grid_HW, align_corners=False):
        N, C, H_in, W_in = img.shape
        if align_corners:
            # 角点对齐模式
            grid = grid / (grid_HW - 1) * 2 -1
        else:
            # 像素中心对齐模式 (默认)
            grid = (grid + 0.5) / grid_HW * 2 - 1

        grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H_out, W_out, 2)
        return F.grid_sample(
            img,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners
        )

    ## 12-tap kernel. f是我们需要采样的原图中的点
    ##    b c
    ##  e f g h
    ##  i j k l
    ##    n o
    def get_near_block(self, src_img, src_x, src_y, channel=1):
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

        sampled_points = []
        for dx, dy in grid_points:
            # Clamp coordinates to valid range
            x_clamped = torch.clamp(src_x + dx, 0, W - 1).long()
            y_clamped = torch.clamp(src_y + dy, 0, H - 1).long()

            output = src_img[:, channel:channel+1, y_clamped, x_clamped]
            sampled_points.append(output)

        return sampled_points

    def gsr_process(self, lr):
        B, C, H, W = lr.shape
        dst_H = int(H * self.scale_factor)
        dst_W = int(W * self.scale_factor)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(dst_H, dtype=lr.dtype, device=self.device),
            torch.arange(dst_W, dtype=lr.dtype, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((x_coords, y_coords), dim=-1)  # (H_out, W_out, 2)
        grid_HW = torch.tensor([dst_W, dst_H], dtype=lr.dtype, device=lr.device)
        pix = self.bilinear_upsample(lr, grid, grid_HW, align_corners=False)

        srcx_f = (x_coords + 0.5) / self.scale_factor - 0.5
        srcy_f = (y_coords + 0.5) / self.scale_factor - 0.5
        src_x = torch.floor(srcx_f)
        src_y = torch.floor(srcy_f)
        fg, gg, jg, kg, bg, cg, eg, hg, ig, lg, ng, og = self.get_near_block(lr, src_x, src_y)
        # shader if edgeVote > edgeThreshold, cal diff, other return pix
        edgeVote = torch.abs(fg - jg) + torch.abs(pix[:, 1:2] - jg) + torch.abs(pix[:, 1:2] - fg)

        mean = (jg + fg + kg + gg) * 0.25
        pix_mean = pix[:, 1:2] - mean
        coff = torch.cat([fg, gg, jg, kg, bg, cg, eg, hg, ig, lg, ng, og], dim=1)
        coff_mean = coff - mean

        u = srcx_f - src_x
        v = srcy_f - src_y
        finalY = self.cal_weight_Y(u, v, coff_mean)
        deltaY = torch.clamp(finalY - pix_mean, -23.0/255.0, 23.0/255.0)
        enhance = torch.clamp(pix + deltaY, 0.0, 1.0)

        mask = edgeVote > self.edgeThreshold
        res = torch.where(mask, enhance, pix)
        return res
