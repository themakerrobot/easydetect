# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the Apache
# License, Version 2.0 — from segment-anything 1.0 (modeling/{common,
# prompt_encoder, mask_decoder, transformer}.py and utils/onnx.py), which
# MobileSAM uses unchanged. Adapted for easydetect (Apache-2.0): only what a
# box prompt needs, one module, a batch of (embedding, box) pairs for
# training, and the ONNX wrapper task="segment" runs.
"""SAM's prompt encoder and mask decoder, the part of MobileSAM that
easydetect fine-tunes on a segmentation dataset of your own. The image
encoder stays as it is (its ONNX file from the mirror); only these 4 M
parameters learn, from embeddings computed once a picture."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMG_SIZE = 1024  # the encoder's input side
EMBED = 256  # prompt / transformer width
GRID = 64  # the image embedding is GRID x GRID
LOW_RES = 256  # masks come out at this side, for the IMG_SIZE frame


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64) -> None:
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> Tensor:
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        return self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1)).permute(2, 0, 1)


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED, image_embedding_size=(GRID, GRID),
                 input_image_size=(IMG_SIZE, IMG_SIZE), mask_in_chans: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, embed_dim) for _ in range(4))
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4), nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans), nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def embed_boxes(self, boxes: Tensor) -> Tensor:
        """Boxes ``(B, 4)`` x1 y1 x2 y2 in the IMG_SIZE frame -> ``(B, 2, C)``."""
        coords = (boxes + 0.5).reshape(-1, 2, 2)
        coords = coords / torch.tensor([self.input_image_size[1], self.input_image_size[0]],
                                       dtype=coords.dtype, device=coords.device)
        corners = self.pe_layer._pe_encoding(coords.float())
        corners[:, 0, :] += self.point_embeddings[2].weight[0]
        corners[:, 1, :] += self.point_embeddings[3].weight[0]
        return corners


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        attn = torch.softmax(q @ k.permute(0, 1, 3, 2) / math.sqrt(q.shape[-1]), dim=-1)
        out = attn @ v
        b, h, n, c = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, n, h * c))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads,
                                                   attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, nn.ReLU)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads,
                                                   attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor,
                key_pe: Tensor) -> tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int = 2, embedding_dim: int = EMBED, num_heads: int = 8,
                 mlp_dim: int = 2048, attention_downsample_rate: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, attention_downsample_rate,
                                 skip_first_layer_pe=(i == 0)) for i in range(depth))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads,
                                                   attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor,
                point_embedding: Tensor) -> tuple[Tensor, Tensor]:
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding,
                                  key_pe=image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q=q, k=k, v=keys))
        return queries, keys


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim], strict=True))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim: int = EMBED, num_multimask_outputs: int = 3,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(2, transformer_dim, 8, 2048)
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4), nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens))
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def predict_masks(self, src: Tensor, pos_src: Tensor,
                      sparse_prompt_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """``src`` and ``pos_src`` already one per prompt, ``(B, C, 64, 64)``
        -> masks ``(B, 4, 256, 256)`` (logits) and their predicted IoU ``(B, 4)``."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).reshape(b, c, h, w)
        upscaled = self.output_upscaling(src)
        hyper_in = torch.stack([mlp(mask_tokens_out[:, i, :]) for i, mlp
                                in enumerate(self.output_hypernetworks_mlps)], dim=1)
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.reshape(b, c, h * w)).reshape(b, -1, h, w)
        return masks, self.iou_prediction_head(iou_token_out)


class SamDecoder(nn.Module):
    """MobileSAM's prompt encoder and mask decoder: ``(image embeddings (B,
    256, 64, 64), boxes (B, 4) in the 1024 frame)`` -> ``(masks (B, 4, 256,
    256) logits, IoU predictions (B, 4))``, one embedding a box. Slot 0 is the
    single-mask output; 1-3 the three a box prompt chooses among."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = MaskDecoder()

    def forward(self, embeddings: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        sparse = self.prompt_encoder.embed_boxes(boxes)
        dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        src = embeddings + dense
        pos = self.prompt_encoder.get_dense_pe().expand(len(boxes), -1, -1, -1)
        return self.mask_decoder.predict_masks(src, pos, sparse)

    def load_mobile_sam(self, state: dict) -> int:
        """Take the prompt encoder and mask decoder of a MobileSAM (or SAM)
        state dict; returns how many tensors were taken."""
        mine = self.state_dict()
        taken = {k: v for k, v in state.items() if k in mine and mine[k].shape == v.shape}
        mine.update(taken)
        self.load_state_dict(mine)
        return len(taken)


class SamDecoderOnnx(nn.Module):
    """What ``decoder.onnx`` computes (SAM's SamOnnxModel with
    return_single_mask): image embeddings ``(1, 256, 64, 64)``, box corners as
    points labelled 2 and 3 ``(N, 2, 2)``, the unused mask input, the picture's
    size -> masks at the picture's size, their predicted IoU, the low-res
    masks. One picture, N boxes, as task="segment" calls it."""

    def __init__(self, decoder: SamDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    @staticmethod
    def resize_longest_image_size(input_image_size: Tensor, longest_side: int) -> Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        return torch.floor(scale * input_image_size + 0.5).to(torch.int64)

    def _embed_points(self, point_coords: Tensor, point_labels: Tensor) -> Tensor:
        pe = self.decoder.prompt_encoder
        point_coords = (point_coords + 0.5) / IMG_SIZE
        point_embedding = pe.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + pe.not_a_point_embed.weight * (point_labels == -1)
        for i in range(pe.num_point_embeddings):
            point_embedding = point_embedding + pe.point_embeddings[i].weight * (point_labels == i)
        return point_embedding

    def forward(self, image_embeddings: Tensor, point_coords: Tensor, point_labels: Tensor,
                mask_input: Tensor, has_mask_input: Tensor, orig_im_size: Tensor):
        pe = self.decoder.prompt_encoder
        sparse = self._embed_points(point_coords, point_labels)
        dense = has_mask_input * pe.mask_downscaling(mask_input) + \
            (1 - has_mask_input) * pe.no_mask_embed.weight.reshape(1, -1, 1, 1)
        n = sparse.shape[0]
        src = torch.repeat_interleave(image_embeddings, n, dim=0) + dense
        pos = torch.repeat_interleave(pe.get_dense_pe(), n, dim=0)
        masks, scores = self.decoder.mask_decoder.predict_masks(src, pos, sparse)
        # a box (two points) takes the best of the three multimask outputs
        reweight = torch.tensor([[1000] + [0] * (self.decoder.mask_decoder.num_mask_tokens - 1)],
                                device=scores.device)
        best = torch.argmax(scores + (point_coords.shape[1] - 2.5) * reweight, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best, :, :].unsqueeze(1)
        scores = scores[torch.arange(masks.shape[0]), best].unsqueeze(1)
        up = F.interpolate(masks, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        pre = self.resize_longest_image_size(orig_im_size, IMG_SIZE)
        up = up[..., :int(pre[0]), :int(pre[1])]
        size = orig_im_size.to(torch.int64)
        up = F.interpolate(up, size=(size[0], size[1]), mode="bilinear", align_corners=False)
        return up, scores, masks
