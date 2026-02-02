import numpy as np
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
from copy import deepcopy
import cv2
from monai import data, transforms
from .dataloaders.data_utils import Resize, PermuteTransform, Normalization
import torch.nn as nn

class IMISPredictor:
    def __init__(self, sam_model):

        super().__init__()
        self.model = sam_model
        self.devices = sam_model.device
        self.reset_image()
 
        if self.model.category_weights is not None:
            self.idx_to_class = self.model.index_to_category 

    def set_image(self,image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ["RGB","BGR",], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            input_image = image[..., ::-1]
        else:
            input_image = image
        # Transform the image to the form expected by the model
        # pixel_mean, pixel_std = np.array((123.675, 116.28, 103.53)), np.array((58.395, 57.12, 57.375))
        # input_image = (image - pixel_mean) / pixel_std

        self.orig_h, self.orig_w, _ = input_image.shape
        self.original_size = (self.orig_h, self.orig_w)
        self.image_size = self.model.image_size
        self.input_h, self.input_w = self.image_size[0], self.image_size[1]

        transform = self.transforms(self.image_size)
        input_image = transform({'image': input_image})['image'][None, :, :, :]
        assert (
            len(input_image.shape) == 4
            and input_image.shape[1] == 3
        ), f"set_torch_image input must be BCHW with long side {self.image_size}."

        self.features = self.model.image_forward(input_image.to(self.devices))
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        text: list = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch, text_torch = None, None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            
            point_coords = self.apply_coords(point_coords, self.original_size, self.image_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.devices)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.devices)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            box = self.apply_boxes(box, self.original_size, self.image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.devices)
            box_torch = box_torch[None, :]

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.devices)
            # Ensure mask_input has correct shape: (1, 1, H, W)
            # Handle different input shapes:
            # - (H, W) -> (1, 1, H, W)
            # - (1, H, W) -> (1, 1, H, W)
            # - (1, 1, H, W) -> (1, 1, H, W) (already correct)
            if mask_input_torch.dim() == 2:
                # (H, W) -> (1, 1, H, W)
                mask_input_torch = mask_input_torch.unsqueeze(0).unsqueeze(0)
            elif mask_input_torch.dim() == 3:
                # (1, H, W) or (C, H, W) -> (1, 1, H, W)
                mask_input_torch = mask_input_torch.unsqueeze(0)
            elif mask_input_torch.dim() != 4:
                raise ValueError(f"mask_input must be 2D, 3D, or 4D tensor, got {mask_input_torch.dim()}D")

        if text is not None:
            text_torch = text #

        masks, low_res_masks, class_list = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            text_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )
        if self.devices == 'cpu':
            masks = masks.detach().numpy()
            low_res_masks = low_res_masks[0].detach().numpy()
        else:
            masks = masks.detach().cpu().numpy()
            low_res_masks = low_res_masks[0].detach().cpu().numpy()

        return masks, low_res_masks, class_list

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        
        prompt = {}
        if text is not None:
            prompt.update({'text_inputs':self.model.text_tokenizer(text).to(self.devices)})
 
        if mask_input is not None:
            prompt.update({'mask_inputs':mask_input})
        
        if point_coords is not None:
            prompt.update({'point_coords':point_coords, 'point_labels':point_labels})

        class_list = []
        masks_list = []

        if boxes is not None:
            for i in range(boxes.shape[1]):
                prompt['bboxes'] = boxes[:,i:i+1,...]
                # Predict masks
                outputs = self.model.forward_decoder(self.features, prompt)
                # Upscale the masks to the original image resolution
                pre_masks = self.postprocess_masks(outputs['masks'], self.original_size)
                masks_list.append(pre_masks)

                if self.model.category_weights:
                    pred_classes = self.predict_category(outputs['semantic_pred'])
                else:
                    pred_classes = 'Sorry, category recognition is currently not supported'
                class_list.append(pred_classes)
          
        else:
            outputs = self.model.forward_decoder(self.features, prompt)
            # Upscale the masks to the original image resolution
            masks_list.append(self.postprocess_masks(outputs['masks'], self.original_size))

            if self.model.category_weights is not None:
                pred_classes = self.predict_category(outputs['semantic_pred'])
            else:
                pred_classes = 'Sorry, category recognition is currently not supported'
            
            class_list.append(pred_classes)

        masks = torch.cat(masks_list, dim=0)

        low_res_masks = outputs['low_res_masks']

        if not return_logits:
            masks = torch.sigmoid(masks)
            masks = (masks > 0.5).float()
        return masks, low_res_masks, class_list

    
    def predict_category(self, semantic_preds):
        logits = nn.functional.normalize(semantic_preds, dim=-1) @ self.model.src_weights
        probs = nn.functional.softmax(logits, dim=-1)
        if self.devices == 'cpu':
            category_preds = int(torch.argmax(probs, dim=-1).squeeze())
        else:
            category_preds = int(torch.argmax(probs, dim=-1).squeeze().cpu())
        return self.idx_to_class[category_preds]
    
    def postprocess_masks(self, masks, original_size):
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)


    def apply_coords_torch(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes, original_size, new_size):
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)


    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features


    def transforms(self, image_size):
        return transforms.Compose(
                            [
                                Resize(keys=["image"], target_size=image_size),  #
                                PermuteTransform(keys=["image"], dims=(2,0,1)),
                                transforms.ToTensord(keys=["image"]),
                                Normalization(keys=["image"]),
                            ]
                        )

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
