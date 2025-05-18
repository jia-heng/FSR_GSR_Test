import torch

EPS = 1e-5
class FSR_EASU(object):
    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transfergray(self, img):
        # B,C,H,W // RGB
        return 0.5 * img[:, 0:1] + img[:, 1:2] + 0.5 * img[:, 2:3]

    ## 12-tap kernel. f 是我们需要采样的原图中的点  zzzz，不需要的点
    ##    z z         z  z
    ##    b c         p0 c
    ##  e f g h   e   f  g  h
    ##  i j k l   p1  j  p2  l
    ##    n o         n  o
    ##    z z         p3 z
    def get_near_block(self, src_img, src_x, src_y):
        B, C, H, W = src_img.shape
        # Generate normalized grid coordinates for all 12 points
        grid_points = [
            # b, c (top row)
            (src_x, src_y - 1),
            (src_x + 1, src_y - 1),

            # e, f, g, h (middle row)
            (src_x - 1, src_y),
            (src_x, src_y),
            (src_x + 1, src_y),
            (src_x + 2, src_y),

            # i, j, k, l (next row)
            (src_x - 1, src_y + 1),
            (src_x, src_y + 1),
            (src_x + 1, src_y + 1),
            (src_x + 2, src_y + 1),

            # n, o (bottom row)
            (src_x, src_y + 2),
            (src_x + 1, src_y + 2)
        ]

        # Sample all points using direct indexing with clamping
        sampled_points = []

        for x_offset, y_offset in grid_points:
            # Clamp coordinates to valid range
            x_clamped = torch.clamp(x_offset, 0, W - 1).long()
            y_clamped = torch.clamp(y_offset, 0, H - 1).long()

            # Initialize output tensor
            output = torch.zeros(B, C, x_clamped.size(0), x_clamped.size(1), device=self.device)
            output[:, :] = src_img[:, :, y_clamped, x_clamped]
            sampled_points.append(output)

        # Unpack to the expected format
        b, c, e, f, g, h, i, j, k, l, n, o = sampled_points

        return b, c, e, f, g, h, i, j, k, l, n, o

    def cal_array_depart(self, left, center, right):
        x_right = torch.abs(right - center)
        x_left = torch.abs(center - left)
        dirx_s = right - left
        return x_right, x_left, dirx_s

    ##    b c         b        c
    ##  e f g h     e f g    f g h     f         g
    ##  i j k l       j        k     i j k     j k l
    ##    n o                          n         o
    def cal_dir_len(self, top, left, center, right, bot):
        x_right, x_left, dirx = self.cal_array_depart(left, center, right)
        max_x= torch.maximum(torch.maximum(x_right, x_left), torch.ones_like(x_right) * EPS)
        lenx = torch.clamp(torch.abs(dirx) / max_x, 0.0, 1.0)
        lenx = lenx * lenx
        y_bot, y_top, diry = self.cal_array_depart(top, center, bot)
        max_y = torch.maximum(torch.maximum(y_bot, y_top), torch.ones_like(y_bot) * EPS)
        leny = torch.clamp(torch.abs(diry) / max_y, 0.0, 1.0)
        leny = leny * leny
        dir = torch.cat([dirx, diry], dim=1)
        length = lenx+leny
        return dir, length

    def cal_color_weight(self, off, dir_x, dir_y, len2, lob, clp):
        # vx = x * cos + y * sin
        # vy = -x * sin + y * cos
        vx = off[0] * dir_x + off[1] * dir_y
        vy = off[0] * (-dir_y) + off[1] * dir_x

        vx = vx * len2[0]
        vy = vy * len2[1]

        d2 = vx * vx + vy * vy
        d2 = torch.clamp(d2, max=clp)

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

    def process(self, lr):
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
        b, c, e, f, g, h, i, j, k, l, n, o = self.get_near_block(lr, src_x, src_y)
        bl, cl, el, fl, gl, hl, il, jl, kl, ll, nl, ol = self.get_near_block(lumen, src_x, src_y)
        u = srcx_f - src_x
        v = srcy_f - src_y

        w_s = (1 - u) * (1 - v)
        dir, length = self.cal_dir_len(bl, el, fl, gl, jl)
        dir_s = dir * w_s
        length_s = length * w_s
        # t (top-right)
        w_t = u * (1 - v)
        dir, length = self.cal_dir_len(cl, fl, gl, hl, kl)
        dir_t = dir * w_t
        length_t = length * w_t
        # u (bottom-left)
        w_u = (1 - u) * v
        dir, length = self.cal_dir_len(fl, il, jl, kl, nl)
        dir_u = dir * w_u
        length_u = length * w_u
        # v (bottom-right)
        w_v = u * v
        dir, length = self.cal_dir_len(gl, jl, kl, ll, ol)
        dir_v = dir * w_v
        length_v = length * w_v

        dir = dir_s + dir_t + dir_u + dir_v
        length = length_s + length_t + length_u + length_v

        dir_x = dir[:, 0]
        dir_y = dir[:, 1]
        # Normalize direction
        dirr = dir_x * dir_x + dir_y * dir_y
        mask = dirr < 1.0 / 32768.0
        dirr[mask] = 1
        dir_x[mask] = 1
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

        temp1 = torch.min(f, g)
        temp2 = torch.min(j, k)
        rgb_min = torch.min(temp1, temp2)
        temp1 = torch.max(f, g)
        temp2 = torch.max(j, k)
        rgb_max = torch.max(temp1, temp2)

        boff = (0.0 - u, -1.0 - v)
        coff = (1.0 - u, -1.0 - v)
        ioff = (-1.0 - u, 1.0 - v)
        joff = (0.0 - u, 1.0 - v)
        foff = (0.0 - u, 0.0 - v)
        eoff = (-1.0 - u, 0.0 - v)
        koff = (1.0 - u, 1.0 - v)
        loff = (2.0 - u, 1.0 - v)
        hoff = (2.0 - u, 0.0 - v)
        goff = (1.0 - u, 0.0 - v)
        ooff = (1.0 - u, 2.0 - v)
        noff = (0.0 - u, 2.0 - v)

        wb = self.cal_color_weight(boff, dir_x, dir_y, len2, lob, clob)
        cb = b*wb
        wc = self.cal_color_weight(coff, dir_x, dir_y, len2, lob, clob)
        cc = c*wc
        wi = self.cal_color_weight(ioff, dir_x, dir_y, len2, lob, clob)
        ci = i*wi
        wj = self.cal_color_weight(joff, dir_x, dir_y, len2, lob, clob)
        cj = j*wj
        wf = self.cal_color_weight(foff, dir_x, dir_y, len2, lob, clob)
        cf = f*wf
        we = self.cal_color_weight(eoff, dir_x, dir_y, len2, lob, clob)
        ce = e*we
        wk = self.cal_color_weight(koff, dir_x, dir_y, len2, lob, clob)
        ck = k*wk
        wl = self.cal_color_weight(loff, dir_x, dir_y, len2, lob, clob)
        cl = l*wl
        wh = self.cal_color_weight(hoff, dir_x, dir_y, len2, lob, clob)
        ch = h*wh
        wg = self.cal_color_weight(goff, dir_x, dir_y, len2, lob, clob)
        cg = g*wg
        wo = self.cal_color_weight(ooff, dir_x, dir_y, len2, lob, clob)
        co = o*wo
        wn = self.cal_color_weight(noff, dir_x, dir_y, len2, lob, clob)
        cn = n*wn
        aw = wb + wc + wi + wj + wf + we + wk + wl + wh + wg + wo + wn
        ac = cb + cc + ci + cj + cf + ce + ck + cl + ch + cg + co + cn
        aw = torch.clamp(aw, min=EPS)
        result = torch.clamp(ac/aw, min=rgb_min, max=rgb_max)

        return result

