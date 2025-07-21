import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SlopeStrengthAnalyzer:
    def __init__(self, device='cuda', eps=1e-5):
        self.device = device
        self.EPS = eps
        self.pi = 3.1415926

        # 存储所有分析结果
        self.slope_values = []
        self.strength_values = []

    def cal_edgeDirection(self, f, g, j, k):
        """计算边缘方向 - 与您的代码完全一致"""
        delta_x = (k + g) - (f + j)
        delta_y = (k + j) - (f + g)

        slope = delta_y / (delta_x + self.EPS)
        slope = slope.atan() / self.pi + 0.5
        strength = torch.sqrt(delta_x * delta_x + delta_y * delta_y)

        return slope, strength

    def get_near_block(self, src_img, src_x, src_y, channel=1):
        """获取邻近像素块"""
        B, C, H, W = src_img.shape
        grid_points = [(0, 0), (1, 0), (0, 1), (1, 1)]  # f, g, j, k

        sampled_points = []
        for dx, dy in grid_points:
            x_clamped = torch.clamp(src_x + dx, 0, W - 1).long()
            y_clamped = torch.clamp(src_y + dy, 0, H - 1).long()
            output = src_img[:, channel:channel + 1, y_clamped, x_clamped]
            sampled_points.append(output)

        return sampled_points

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

    def analyze_single_image(self, image_path, scale_factor=1.5):
        """分析单张图片的slope和strength分布"""
        # 读取图片
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片: {image_path}")
                return None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path

        # 转换为tensor
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        img_tensor = img_tensor.to(self.device)

        B, C, H, W = img_tensor.shape
        dst_H = int(H * scale_factor)
        dst_W = int(W * scale_factor)

        # 创建目标坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(dst_H, dtype=img_tensor.dtype, device=self.device),
            torch.arange(dst_W, dtype=img_tensor.dtype, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((x_coords, y_coords), dim=-1)  # (H_out, W_out, 2)
        WH = torch.tensor([dst_W, dst_H], device=self.device)
        pix = self.bilinear_upsample(img_tensor, grid, WH, align_corners=False)
        # 计算源坐标
        srcx_f = (x_coords + 0.5) / scale_factor - 0.5
        srcy_f = (y_coords + 0.5) / scale_factor - 0.5
        src_x = torch.floor(srcx_f)
        src_y = torch.floor(srcy_f)

        # 获取邻近像素块（使用绿色通道）
        fg, gg, jg, kg = self.get_near_block(img_tensor, src_x, src_y, channel=1)
        edgeVote = torch.abs(fg - jg) + torch.abs(pix[:, 1:2] - jg) + torch.abs(pix[:, 1:2] - fg)
        edge_mask = edgeVote > 8.0 / 255.0
        # 计算slope和strength
        slope, strength = self.cal_edgeDirection(fg, gg, jg, kg)
        slope_masked = slope[edge_mask]
        strength_masked = strength[edge_mask]
        # 转换为numpy数组并收集数据
        slope_np = slope_masked.cpu().numpy().flatten()
        strength_np = strength_masked.cpu().numpy().flatten()

        # 过滤异常值
        valid_mask = np.isfinite(slope_np) & np.isfinite(strength_np)
        slope_np = slope_np[valid_mask]
        strength_np = strength_np[valid_mask]

        return slope_np, strength_np

    def analyze_multiple_images(self, image_paths, scale_factor=1.5):
        """分析多张图片"""
        print(f"开始分析 {len(image_paths)} 张图片...")

        for i, img_path in enumerate(tqdm(image_paths, desc="分析图片")):
            slope_vals, strength_vals = self.analyze_single_image(img_path, scale_factor)

            if slope_vals is not None and strength_vals is not None:
                self.slope_values.extend(slope_vals)
                self.strength_values.extend(strength_vals)

                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1} 张图片，当前数据量: {len(self.slope_values)}")

    def analyze_directory(self, directory_path, scale_factor=1.5, extensions=None):
        """分析目录中的所有图片"""
        if extensions is None:
            extensions = ['.jpg', '.png', '.bmp']

        image_paths = []
        for ext in extensions:
            image_paths.extend([
                os.path.join(directory_path, f)
                for f in os.listdir(directory_path)
                if f.lower().endswith(ext.lower())
            ])

        print(f"找到 {len(image_paths)} 张图片")
        self.analyze_multiple_images(image_paths, scale_factor)

    def get_statistics(self):
        """获取统计信息"""
        if not self.slope_values or not self.strength_values:
            print("没有数据可以分析")
            return None

        slope_array = np.array(self.slope_values)
        strength_array = np.array(self.strength_values)

        stats = {
            'slope': {
                'min': np.min(slope_array),
                'max': np.max(slope_array),
                'mean': np.mean(slope_array),
                'median': np.median(slope_array),
                'std': np.std(slope_array),
                'percentiles': np.percentile(slope_array, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            },
            'strength': {
                'min': np.min(strength_array),
                'max': np.max(strength_array),
                'mean': np.mean(strength_array),
                'median': np.median(strength_array),
                'std': np.std(strength_array),
                'percentiles': np.percentile(strength_array, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            }
        }

        return stats

    def plot_distributions(self, bins=1000, save_path=None):
        """绘制分布图"""
        if not self.slope_values or not self.strength_values:
            print("没有数据可以绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Slope分布
        axes[0, 0].hist(self.slope_values, bins=bins, alpha=0.7, color='blue', density=True)
        axes[0, 0].set_title('Slope Distribution')
        axes[0, 0].set_xlabel('Slope Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)

        # Strength分布
        axes[0, 1].hist(self.strength_values, bins=bins, alpha=0.7, color='red', density=True)
        axes[0, 1].set_title('Strength Distribution')
        axes[0, 1].set_xlabel('Strength Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)

        # Slope累积分布
        slope_sorted = np.sort(self.slope_values)
        axes[1, 0].plot(slope_sorted, np.linspace(0, 1, len(slope_sorted)), color='blue')
        axes[1, 0].set_title('Slope Cumulative Distribution')
        axes[1, 0].set_xlabel('Slope Value')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)

        # Strength累积分布
        strength_sorted = np.sort(self.strength_values)
        axes[1, 1].plot(strength_sorted, np.linspace(0, 1, len(strength_sorted)), color='red')
        axes[1, 1].set_title('Strength Cumulative Distribution')
        axes[1, 1].set_xlabel('Strength Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_quantization_levels(self, num_levels=16, method='equal_frequency'):
        """生成量化级别"""
        if not self.slope_values or not self.strength_values:
            print("没有数据可以生成量化级别")
            return None, None

        slope_array = np.array(self.slope_values)
        strength_array = np.array(self.strength_values)

        if method == 'equal_frequency':
            # 等频率划分
            slope_levels = np.percentile(slope_array, np.linspace(0, 100, num_levels + 1))
            strength_levels = np.percentile(strength_array, np.linspace(0, 100, num_levels + 1))
        elif method == 'equal_width':
            # 等宽度划分
            slope_levels = np.linspace(np.min(slope_array), np.max(slope_array), num_levels + 1)
            strength_levels = np.linspace(np.min(strength_array), np.max(strength_array), num_levels + 1)
        else:
            raise ValueError("method must be 'equal_frequency' or 'equal_width'")

        return slope_levels, strength_levels

    def save_analysis_results(self, save_path):
        """保存分析结果"""
        results = {
            'slope_values': self.slope_values,
            'strength_values': self.strength_values,
            'statistics': self.get_statistics()
        }

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"分析结果已保存到: {save_path}")

    def load_analysis_results(self, load_path):
        """加载分析结果"""
        with open(load_path, 'rb') as f:
            results = pickle.load(f)

        self.slope_values = results['slope_values']
        self.strength_values = results['strength_values']

        print(f"分析结果已从 {load_path} 加载")
        return results['statistics']


# 使用示例
def main():
    img_path = r'D:\s00827220\projects\FSR_GSR_Test\test\40_png'
    # 创建分析器
    analyzer = SlopeStrengthAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu')

    # 分析图片目录
    analyzer.analyze_directory(img_path, scale_factor=1.5)

    # 或者分析单张图片
    # slope_vals, strength_vals = analyzer.analyze_single_image('/path/to/image.jpg')

    # 获取统计信息
    stats = analyzer.get_statistics()
    print("统计信息:")
    for key, value in stats.items():
        print(f"{key}:")
        for stat_name, stat_value in value.items():
            print(f"  {stat_name}: {stat_value}")

    # 绘制分布图
    analyzer.plot_distributions(bins=1000, save_path='distribution_plot.png')

    # 生成量化级别
    slope_levels, strength_levels = analyzer.generate_quantization_levels(num_levels=16, method='equal_frequency')
    print(f"Slope量化级别: {slope_levels}")
    print(f"Strength量化级别: {strength_levels}")

    # 保存结果
    analyzer.save_analysis_results('analysis_results.pkl')


if __name__ == "__main__":
    main()