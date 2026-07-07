# Original code src: https://github.com/overlappredator/OverlapPredator/blob/main/models/gcn.py

from copy import deepcopy
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


def get_graph_feature(
    coords: torch.Tensor, feats: torch.Tensor, k: int = 9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized vanilla KNN with improved memory efficiency:
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        coords:     [B, 2, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
        idx:        [B, N, k]
    """
    # import pdb; pdb.set_trace()
    B, C, N = feats.shape
    k = min(k, N - 1)
    device = coords.device
    
    # Cache eye matrix to avoid recomputation
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    
    # Compute pairwise squared distances using torch.cdist (optimized)
    coords_t = coords.transpose(1, 2)  # [B, N, 3]
    dist_sq = torch.cdist(coords_t, coords_t, p=2.0).square()
    
    # Mask self-connections and get k-nearest neighbors
    dist_sq.masked_fill_(eye, float('inf'))
    idx = dist_sq.topk(k=k, dim=-1, largest=False, sorted=True).indices
    
    # Optimized feature gathering using advanced indexing
    idx_flat = idx.view(B, -1)  # [B, N*k]
    feats_gathered = feats.gather(2, idx_flat.unsqueeze(1).expand(-1, C, -1))  # [B, C, N*k]
    neigh_feats = feats_gathered.view(B, C, N, k)  # [B, C, N, k]
    
    # Compute feature differences efficiently
    feats_expanded = feats.unsqueeze(-1)  # [B, C, N, 1]
    feat_diff = neigh_feats - feats_expanded  # [B, C, N, k]
    feats_cat = torch.cat([feats_expanded.expand(-1, -1, -1, k), feat_diff], dim=1)
    
    return feats_cat, idx

def geometric_embedding(coords_center, coords_neighbor):
    """
    Compute geometric embedding with both raw difference and normalized direction.

    Input:
        coords_center: [B, 2, N, k] - center point coordinates
        coords_neighbor: [B, 2, N, k] - neighbor point coordinates

    Returns:
        embedding: [B, 4, N, k] - concatenation of:
            - raw difference (2 channels): preserves scale information
            - normalized direction (2 channels): scale-invariant directional info
    """
    # Ensure the tensors are float for calculations
    coords_center = coords_center.float()
    coords_neighbor = coords_neighbor.float()

    # Calculate raw coordinate difference
    diff = coords_neighbor - coords_center  # [B, 2, N, k]

    # Calculate distance (L2 norm)
    dist = torch.norm(diff, dim=1, keepdim=True)  # [B, 1, N, k]

    # Calculate normalized direction vector
    direction = diff / (dist + 1e-8)  # [B, 2, N, k], add epsilon to avoid division by zero

    # Concatenate raw difference and normalized direction
    return torch.cat([diff, direction], dim=1)  # [B, 4, N, k]

class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, k: int = 10, in_channel: int = 128, num_registers: int = 4) -> None:
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(
            feature_dim * 2, feature_dim * 2, kernel_size=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.conv3_old = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3_old = nn.InstanceNorm2d(feature_dim)

        self.in_channel = in_channel
        self.annuconv1 = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        self.annuconv2 = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        # angle encoder (updated to handle 4 channels: diff_x, diff_y, dir_x, dir_y)
        self.angle_enc1 = nn.Sequential(
                nn.Conv2d(4, self.in_channel, (1, 3), stride=(1, 3)),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

        # Geometry encoders for old path (maxpooling path)
        self.geo_enc_old1 = nn.Sequential(
                nn.Conv2d(4, feature_dim, kernel_size=1, bias=False),
                nn.InstanceNorm2d(feature_dim),
                nn.ReLU(inplace=True),
            )
        self.geo_enc_old2 = nn.Sequential(
                nn.Conv2d(4, feature_dim * 2, kernel_size=1, bias=False),
                nn.InstanceNorm2d(feature_dim * 2),
                nn.ReLU(inplace=True),
            )

        self.k = k
        self.num_registers = num_registers
        self.feature_dim = feature_dim

        # Register tokens - learnable global information storage
        self.register_tokens = nn.Parameter(torch.randn(1, feature_dim, num_registers) * 0.02)

        # Normalization for registers
        self.register_norm = nn.LayerNorm(feature_dim)

        # Attention mechanisms for register communication
        self.register_self_attn = MultiHeadedAttention(4, feature_dim)
        self.spatial_to_register = MultiHeadedAttention(4, feature_dim)
        self.register_to_spatial = MultiHeadedAttention(4, feature_dim)

        # MLPs for processing register interactions
        self.register_update_mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        self.spatial_enhance_mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])

        # Learnable combination weights for register updates
        self.register_gate = nn.Parameter(torch.ones(1))
        self.spatial_gate = nn.Parameter(torch.ones(1))

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N] - enhanced spatial features
            registers:  [B, C, num_registers] - updated register tokens
        """
        B, C, N = features.size()

        # PHASE 1: LOCAL INFORMATION GATHERING - NEW PATH (annular-angle convolution)
        x0_new = features.unsqueeze(-1)  # [B, C, N, 1]
        x1_old, x1_old_ind = get_graph_feature(coords, features, self.k)

        # first update
        x1 = self.annuconv1(x1_old)

        # angle embedding
        coords1 = coords.unsqueeze(2).repeat(1, 1, N, 1)
        nei1 = torch.gather(coords1, dim=-1, index=x1_old_ind.unsqueeze(1).repeat(1, 2, 1, 1))
        ang1 = geometric_embedding(coords.unsqueeze(-1).repeat(1, 1, 1, self.k), nei1)
        f_ang1 = self.angle_enc1(ang1)

        # second update
        x2, x2_ind = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = self.annuconv2(x2)

        # ang++
        nei2 = torch.gather(coords1, dim=-1, index=x2_ind.unsqueeze(1).repeat(1, 2, 1, 1))
        ang2 = geometric_embedding(coords.unsqueeze(-1).repeat(1, 1, 1, self.k), nei2)
        f_ang2 = self.angle_enc1(ang2)

        # final update (new path)
        x3_new = torch.cat((x0_new, x1 + f_ang1, x2 + f_ang2), dim=1)
        x3_new = F.leaky_relu(self.in3(self.conv3(x3_new)), negative_slope=0.2).view(B, -1, N)

        # PHASE 1: LOCAL INFORMATION GATHERING - OLD PATH (maxpooling with geometry)
        x0_old = features.unsqueeze(-1)  # [B, C, N, 1]
        x1_old_graph, x1_old_ind = get_graph_feature(coords, features, self.k)
        x1_old = F.leaky_relu(self.in1(self.conv1(x1_old_graph)), negative_slope=0.2)

        # Add geometry embedding for first layer
        coords1_old = coords.unsqueeze(2).repeat(1, 1, N, 1)
        nei1_old = torch.gather(coords1_old, dim=-1, index=x1_old_ind.unsqueeze(1).repeat(1, 2, 1, 1))
        geo1_old = geometric_embedding(coords.unsqueeze(-1).repeat(1, 1, 1, self.k), nei1_old)
        f_geo1_old = self.geo_enc_old1(geo1_old)

        # Geometry-modulated features before maxpooling
        x1_old = x1_old + f_geo1_old
        x1_old = x1_old.max(dim=-1, keepdim=True)[0]

        x2_old, x2_old_ind = get_graph_feature(coords, x1_old.squeeze(-1), self.k)
        x2_old = F.leaky_relu(self.in2(self.conv2(x2_old)), negative_slope=0.2)

        # Add geometry embedding for second layer
        nei2_old = torch.gather(coords1_old, dim=-1, index=x2_old_ind.unsqueeze(1).repeat(1, 2, 1, 1))
        geo2_old = geometric_embedding(coords.unsqueeze(-1).repeat(1, 1, 1, self.k), nei2_old)
        f_geo2_old = self.geo_enc_old2(geo2_old)

        # Geometry-modulated features before maxpooling
        x2_old = x2_old + f_geo2_old
        x2_old = x2_old.max(dim=-1, keepdim=True)[0]

        x3_old = torch.cat((x0_old, x1_old, x2_old), dim=1)
        x3_old = F.leaky_relu(self.in3_old(self.conv3_old(x3_old)), negative_slope=0.2).view(B, -1, N)

        # Combine both local paths
        local_features = x3_new + x3_old

        # PHASE 2: GLOBAL INFORMATION GATHERING with registers
        # Initialize registers
        registers = self.register_tokens.expand(B, -1, -1)  # [B, C, num_registers]
        registers = self.register_norm(registers.transpose(-1, -2)).transpose(-1, -2)

        # Registers collect global information from local processed features
        register_messages = self.spatial_to_register(registers, local_features, local_features)
        registers = registers + self.register_gate * self.register_update_mlp(torch.cat([registers, register_messages], dim=1))

        # Register self-attention for internal communication
        register_self_msg = self.register_self_attn(registers, registers, registers)
        registers = registers + register_self_msg
        registers = self.register_norm(registers.transpose(-1, -2)).transpose(-1, -2)

        # Spatial features get enhanced global context from registers
        spatial_context = self.register_to_spatial(local_features, registers, registers)
        output_features = local_features + self.spatial_gate * spatial_context

        # PHASE 3: Final register update with output features
        final_register_messages = self.spatial_to_register(registers, output_features, output_features)
        registers = registers + self.register_gate * self.register_update_mlp(torch.cat([registers, final_register_messages], dim=1))

        # Final register self-attention for consolidation
        final_register_self_msg = self.register_self_attn(registers, registers, registers)
        registers = registers + final_register_self_msg
        registers = self.register_norm(registers.transpose(-1, -2)).transpose(-1, -2)

        return output_features, registers


def MLP(channels: Sequence[int], do_bn: bool = True) -> nn.Sequential:
    """Multi-layer perceptron"""
    n = len(channels)
    layers: List[nn.Module] = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class SCAttention(nn.Module):
    """Predator + SuperGlue Self-Cross Attention Implementation with Register Nodes"""

    def __init__(
        self,
        layer_names: Iterable[str], # self,cross,self
        num_head: int = 4,
        feature_dim: int = 128,
        k: int = 9,
        num_registers: int = 4,
    ) -> None:
        super().__init__()
        self.names = layer_names
        self.num_registers = num_registers
        self.feature_dim = feature_dim
        layers: List[nn.Module] = []
        for atten_type in layer_names:
            if atten_type == "self":
                layers.append(SelfAttention(feature_dim, k, num_registers=num_registers))
            else:
                layers.append(AttentionalPropagation(feature_dim, num_head))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        coords0: torch.Tensor,
        coords1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: descs [B, C, N] *coords: [B, D, N]

        # Initialize registers as None - they'll be created in first self-attention layer
        registers0 = None
        registers1 = None

        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                # Cross-attention: treat registers as additional normal nodes
                # if registers0 is not None and registers1 is not None:
                    # Concatenate registers with spatial features
                desc0_with_reg = torch.cat([desc0, registers0], dim=-1)  # [B, C, N+num_reg]
                desc1_with_reg = torch.cat([desc1, registers1], dim=-1)  # [B, C, M+num_reg]

                # Do normal cross-attention on combined features
                enhanced0_with_reg = layer(desc0_with_reg, desc1_with_reg)
                enhanced1_with_reg = layer(desc1_with_reg, desc0_with_reg)

                # Split back into spatial and register parts
                N0 = desc0.size(-1)
                N1 = desc1.size(-1)
                desc0 = desc0 + enhanced0_with_reg[:, :, :N0]
                registers0 = enhanced0_with_reg[:, :, N0:]
                desc1 = desc1 + enhanced1_with_reg[:, :, :N1]
                registers1 = enhanced1_with_reg[:, :, N1:]
                # else:
                #     # No registers yet, do normal cross-attention
                #     desc0 = desc0 + layer(desc0, desc1)
                #     desc1 = desc1 + layer(desc1, desc0)

            elif name == "self":
                # Self-attention creates/updates registers
                desc0, registers0 = layer(coords0, desc0)
                desc1, registers1 = layer(coords1, desc1)

        return desc0, desc1
