import numpy as np
import torch
from skimage import filters

from type import imgf32


class GRConvNet:
    def __init__(self, model_path, device, apply_gaussian):
        self.size = 300  # 当前网络只接收这一尺寸的输入

        self.apply_gaussian = apply_gaussian
        self.device = device
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        unlazy = torch.zeros((1, 4, 300, 300)).to(self.device)
        self.model(unlazy)  # 网络第一次推理比较慢

    def post_process_output(
        self, q_img, cos_img, sin_img, width_img, apply_gaussian=True
    ):
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * 100.0
        if apply_gaussian:
            q_img = filters.gaussian(q_img, 2.0, preserve_range=True)
            ang_img = filters.gaussian(ang_img, 2.0, preserve_range=True)
            width_img = filters.gaussian(width_img, 1.0, preserve_range=True)
        return q_img, ang_img, width_img

    def predict(
        self, rgb_img: imgf32, depth_img: imgf32
    ) -> tuple[imgf32, imgf32, imgf32]:
        """
        Args:
            rgb_img: (3, 300, 300) 归一化后的RGB图像
            depth_img: (1, 300, 300) 归一化后的深度图像
        Returns:
            q_img, ang_img, width_img: (300,300) 逐像素预测的抓取质量,角度和宽度
        """
        assert rgb_img.shape == (3, 300, 300)
        assert depth_img.shape == (1, 300, 300)
        x = np.concatenate((depth_img, rgb_img), 0)
        x = np.expand_dims(x, 0)  # (1, 4, size, size)
        x = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            pred = self.model.predict(x)

        q_img, ang_img, width_img = self.post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"], self.apply_gaussian
        )

        return q_img, ang_img, width_img
