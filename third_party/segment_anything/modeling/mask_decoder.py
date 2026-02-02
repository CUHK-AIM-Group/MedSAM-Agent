# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type,  Optional

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        semantic_out_dim: int = 512,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.semantic_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.text_output = nn.Linear(transformer_dim, transformer_dim // 8)
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)  #256 256 4 3
        self.semantic_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, semantic_out_dim, iou_head_depth)  #256 256 4 3

    def forward(
        self,
        image_embeddings: torch.Tensor,   #[B, 256, 64, 64]
        image_pe: torch.Tensor,           #[1, 256, 64, 64]
        sparse_prompt_embeddings: torch.Tensor, #[B, 3, 256]
        dense_prompt_embeddings: torch.Tensor,  #[B, 256, 64, 64]
        text_prompt_embeddings: Optional[torch.Tensor], #[B, 256]
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        masks, iou_pred, semantic_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            text_prompt_embeddings=text_prompt_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        semantic_pred = semantic_pred[:, mask_slice, :]

        outputs = {'low_res_masks':masks, 'iou_pred':iou_pred, 'semantic_pred':semantic_pred}
        return outputs

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        text_prompt_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.semantic_tokens.weight], dim=0)  #iou_token:[1,256]  mask_tokens:[4,256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #b,1,5,256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        num_tokens = tokens.shape[1]
        sparse_dim = sparse_prompt_embeddings.shape[1]
  
        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        semantic_tokens_out = hs[:, (1 + self.num_mask_tokens):(num_tokens-sparse_dim), :]
   
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  #[1,4,32]

        b, c, h, w = upscaled_embedding.shape  #[1, 32, 256, 256]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        if text_prompt_embeddings is not None:
            upscaled_embedding = upscaled_embedding.view(b, c, h * w)
            text_prompt_embeddings = self.text_output(text_prompt_embeddings.unsqueeze(1))
            text_mask = (text_prompt_embeddings @ upscaled_embedding).view(b, -1, h, w)
            text_mask = text_mask.repeat(1, masks.shape[1], 1, 1)
            masks = masks + text_mask
            
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        semantic_pred = self.semantic_prediction_head(semantic_tokens_out)
        return masks, iou_pred, semantic_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x) # source code
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
