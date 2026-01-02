import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _get_cam_weights(self, grads):
        return torch.mean(grads, dim=[2, 3], keepdim=True)

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError()

        weights = self._get_cam_weights(self.gradients)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize the CAM
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam, class_idx

def generate_cam_image(model, target_layer, input_tensor, class_idx=None):
    grad_cam = GradCAM(model, target_layer)
    cam, class_idx = grad_cam(input_tensor.unsqueeze(0))
    
    cam_pil = to_pil_image(cam.squeeze(0))
    
    original_pil = to_pil_image(input_tensor)

    cam_resized = cam_pil.resize(original_pil.size, Image.BICUBIC)
    
    cam_heatmap = np.array(cam_resized)
    cam_heatmap = (255 * cam_heatmap).astype(np.uint8)

    # Apply colormap
    import matplotlib.cm as cm
    heatmap = cm.jet(cam_heatmap)[:, :, :3] * 255
    heatmap_pil = Image.fromarray(heatmap.astype(np.uint8))

    # Superimpose heatmap on original image
    overlay = Image.blend(original_pil, heatmap_pil, alpha=0.5)

    return overlay, class_idx
