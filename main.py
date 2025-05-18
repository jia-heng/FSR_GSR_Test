import cv2
import numpy as np
import torch
import torch.nn.functional as F
from easu_torch import FSR_EASU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    image = cv2.imread("mid.png", cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    output_path = "./test/mid_fsr_x2.png"
    fsr = FSR_EASU()
    hr_img = fsr.process(img_tensor)
    hr_rgb = (hr_img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    hr_bgr = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, hr_bgr)

    bil_output_path = "./test/mid_bil_x2.png"
    bil_tensor = F.interpolate(img_tensor, scale_factor=2.0, mode="bilinear")
    bil_hr_rgb = (bil_tensor[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    bil_hr_bgr = cv2.cvtColor(bil_hr_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(bil_output_path, bil_hr_bgr)
    
