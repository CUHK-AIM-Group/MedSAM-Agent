
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from monai import data, transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from skimage.measure import label, regionprops

class Resize(transforms.Transform):
    def __init__(self, keys, target_size):
        self.keys = keys
        self.target_size = target_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if len(d[key].shape) == 4:
                label = d[key]
                resized_labels = np.zeros((label.shape[0], self.target_size[0], self.target_size[1]))
                for i in range(label.shape[0]):
                    pil_label = to_pil_image(label[i])
                    resized_label = resize(pil_label, self.target_size, interpolation=InterpolationMode.NEAREST)
                    resized_labels[i] = np.array(resized_label)
                d[key] = resized_labels
            else:
                image = to_pil_image(d[key])
                d[key] = resize(image, self.target_size, interpolation=InterpolationMode.NEAREST)
                d[key] = np.array(d[key])
            
            if len(d[key].shape) == 2:
                d[key] = d[key][np.newaxis, ...]
        return d

class PermuteTransform(transforms.Transform):
    def __init__(self, keys, dims):
        self.dims = dims
        self.keys = keys
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.transpose(d[key], self.dims)
        return d

class LongestSidePadding(transforms.Transform):
    def __init__(self, keys, input_size):
        self.keys = keys
        self.input_size = input_size
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            h, w = d[key].shape[-2:]
            padh = self.input_size - h
            padw = self.input_size - w
            d[key] = F.pad(d[key], (0, padw, 0, padh))
        return d

class Normalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
        pixel_mean = (123.675, 116.28, 103.53)
        pixel_std = (58.395, 57.12, 57.375)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] - self.pixel_mean) / self.pixel_std
        return d
    

def get_points_from_mask(mask, get_point=1, top_num=0.5):

    if len(mask.shape) > 2:
        mask = mask.squeeze()

    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    centroid = np.mean(fg_coords, axis=0)  
    distances = np.sqrt(np.sum((fg_coords - centroid)**2, axis=1))  
    sorted_indices = np.argsort(distances) 
    num_points = len(fg_coords)  
    top_k = int(num_points * top_num) 
    top_indices = sorted_indices[:top_k]  

    try:
        coord = fg_coords[np.random.choice(top_indices, size=get_point)[0]]
        label = 1

    except:
        coord = bg_coords[np.random.randint(len(bg_coords))]
        label = 0

    coords, labels = torch.as_tensor([coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)

    return coords, labels


def get_bboxes_from_mask(masks, offset=0):
    if masks.size(1) == 1:
        masks = masks.squeeze(1)
    B, H, W = masks.shape
    bounding_boxes = []
    for i in range(B):
        mask = masks[i]
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            bounding_boxes.append((0, 0, 0, 0))
        else:
            y0, y1 = y_coords.min().item(), y_coords.max().item()
            x0, x1 = x_coords.min().item(), x_coords.max().item()

            if offset > 0:
                y0 = max(0, y0 + torch.randint(-offset, offset + 1, (1,)).item())
                y1 = min(W - 1, y1 + torch.randint(-offset, offset + 1, (1,)).item())
                x0 = max(0, x0 + torch.randint(-offset, offset + 1, (1,)).item())
                x1 = min(H - 1, x1 + torch.randint(-offset, offset + 1, (1,)).item())

            bounding_boxes.append((x0, y0, x1, y1))

    return torch.tensor(bounding_boxes, dtype=torch.float).unsqueeze(1)
    

if __name__ == '__main__':
    image_path = r'demo_image'
    import cv2
    import matplotlib.pyplot as plt 
    import numpy as np
    masks = cv2.imread(image_path, 0) // 255.
    bboxes = get_bboxes_from_mask(torch.tensor(masks).unsqueeze(0), offset=0)  
    bboxes = bboxes.squeeze(0).numpy()  
    plt.imshow(masks, cmap='gray')  
    for box in bboxes:  
        x0, y0, x1, y1 = box  
        plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='r', facecolor='none'))  
  
    plt.show()
