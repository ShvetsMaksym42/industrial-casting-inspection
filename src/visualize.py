import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def heatmap(image, save_path, model, target_layer):
    with torch.enable_grad():
        rgb_image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        rgb_image = (rgb_image*[0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1).astype(np.float32)
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [BinaryClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=image, targets=targets)[0, :]
        grayscale_cam = 1.0 - grayscale_cam
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        original_image = np.uint8(255 * cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        heatmap_image = np.uint8(255 * cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        final_image = cv2.hconcat([original_image, heatmap_image])

        cv2.imwrite(save_path, final_image)