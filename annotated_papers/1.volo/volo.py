import math
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from pic.model import register_model


class OutLookAttn(nn.Module):
    """Outlook Attention

    modification from paper (copied from author code):
    - apply stride(2) option to reduce flops (maybe increase performance also)
    """

    def __init__(self, dim, head, H, W, K=3, padding=1, stride=2, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Conv2d(dim, head * K ** 4, 1)
        self.proj = nn.Linear(dim, dim)
        self.unfold = nn.Unfold(K, padding=padding, stride=stride)
        self.fold = nn.Fold((H, W), K, padding=padding, stride=stride)
        self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.attn_drop = nn.Dropout(attn_drop)

        self.K = K
        self.H = H
        self.W = W
        self.head = head
        self.dim = dim

        self.scale = (dim / head) ** -0.5
        self.seq_len = (H // stride) * (H // stride)

    def forward(self, x):  # input(x): (B, H * W, dim)
        # value(v): (B, H * W, dim) -> (B, dim, H, W) -> (B, H * W // stride ** 2, head, K**2, dim / head)
        v = self.v_pj(x).permute(0, 2, 1).reshape(-1, self.dim, self.H, self.W)
        v = self.unfold(v).reshape(-1, self.head, self.dim // self.head, self.K ** 2, self.seq_len)
        v = v.permute(0, 4, 1, 3, 2)

        # attention(a): (B, H * W, dim) -> (B, H * W // stride ** 2, head, K ** 2, K ** 2)
        a = self.attn(self.avg_pool(x.permute(0, 2, 1).reshape(-1, self.dim, self.H, self.W)))
        a = a.permute(0, 2, 3, 1).reshape(-1, self.seq_len, self.head, self.K ** 2, self.K ** 2)
        a = F.softmax(a * self.scale, dim=-1)
        a = self.attn_drop(a)

        x = (a @ v).permute(0, 2, 4, 3, 1).reshape(-1, self.dim * (self.K ** 2), self.seq_len)
        x = self.fold(x).permute(0, 2, 3, 1).reshape(-1, self.H * self.W, self.dim)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class Outlooker(nn.Module):
    """Outlook Attention + MLP

    details:
    1. qkv_bias = False
    2. attention dropout = 0.0
    """

    def __init__(self, dim, head, mlp_ratio, H, W, K=3, padding=1, stride=2, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.outlook_attn = OutLookAttn(dim, head, H, W, K, padding, stride, qkv_bias, attn_drop)
        self.mlp = MLP(dim, mlp_ratio)
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x_hat = self.outlook_attn(self.ln1(x)) + x
        z = self.mlp(self.ln2(x_hat)) + x_hat

        return z


class ConvNormAct(nn.Sequential):
    def __init__(self, in_dim, out_dim, kernel_size, padding, stride):
        super().__init__(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))


class PatchEmbedding(nn.Module):
    """Non-overlapping embedding function

    difference from paper (copied from author repo)
    - use 4 conv layer in first patch embedding (img -> (pe) -> stage1)
    - add positional embedding in second patch embedding (stage1 -> (pe) -> stage2)
    """

    def __init__(self, in_dim, out_dim, H, W, patch_size, use_stem=False, hidden_dim=64, add_pe=False):
        super().__init__()
        if use_stem:
            self.conv = nn.Sequential(
                ConvNormAct(in_dim, hidden_dim, kernel_size=7, padding=3, stride=2),
                ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
                ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(hidden_dim, out_dim, patch_size // 2, patch_size // 2)
            )
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, patch_size, patch_size)

        self.patch_len = H * W // (patch_size * patch_size)
        self.H = H
        self.W = W
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.add_pe = add_pe
        if self.add_pe:
            self.pe = nn.Parameter(torch.zeros([1, self.patch_len, self.out_dim]))

    def forward(self, x):
        if x.ndim != 4:
            x = x.permute(0, 2, 1).reshape(-1, self.in_dim, self.H, self.W)
        x = self.conv(x).permute(0, 2, 3, 1).reshape(-1, self.patch_len, self.out_dim)

        if self.add_pe:
            x = x + self.pe.expand(x.size(0), -1, -1)

        return x


class MHSA(nn.Module):
    def __init__(self, dim, head, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.k = dim // head
        self.div = math.sqrt(self.k)
        self.head = head
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = [out.reshape(B, N, self.head, self.k).permute(0, 2, 1, 3) for out in
                   self.qkv(x).tensor_split(3, dim=-1)]

        attn = q @ k.transpose(-1, -2) / self.div
        attn_prob = F.softmax(attn, dim=-1)
        attn_prob = self.attn_drop(attn_prob)

        out = attn_prob @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.proj(out)

        return out


class SelfAttention(nn.Module):
    """Self Attention

    Details: drop_path_rate is only applied to this module
    """

    def __init__(self, dim, mlp_ratio, head, qkv_bias=False, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.attn = MHSA(dim, head, qkv_bias, attn_drop)
        self.mlp = MLP(dim, mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(self.norm1(x))) + x
        x = self.drop_path(self.mlp(self.norm2(x))) + x

        return x


class MHCA(nn.Module):
    def __init__(self, dim, head, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.k = dim // head
        self.div = math.sqrt(self.k)
        self.head = head
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, D = x.shape
        q = self.q(x[:, :1]).reshape(B, 1, self.head, self.k).permute(0, 2, 1, 3)
        k, v = [out.reshape(B, N - 1, self.head, self.k).permute(0, 2, 1, 3) for out in
                self.kv(x[:, 1:]).tensor_split(2, dim=-1)]

        attn = q @ k.transpose(-1, -2) / self.div
        attn_prob = F.softmax(attn, dim=-1)
        attn_prob = self.attn_drop(attn_prob)

        out = attn_prob @ v
        out = out.permute(0, 2, 1, 3).reshape(B, 1, D)
        out = self.proj(out)

        return out


class ClassAttention(nn.Module):
    """Class Attention"""

    def __init__(self, dim, mlp_ratio, head, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.attn = MHCA(dim, head, qkv_bias, attn_drop)
        self.mlp = MLP(dim, mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, cls_x):
        cls, x = cls_x
        z = torch.concat([cls, x], dim=1)
        cls = self.attn(self.norm1(z)) + cls
        cls = self.mlp(self.norm2(cls)) + cls

        return cls, x


model_config = {
    'volo_d1_224': {'parameter': dict(H=224, W=224,
                                      s1_num=4, s1_dim=192, s1_head=6, s1_mlp_ratio=3,
                                      s2_num=14, s2_dim=384, s2_head=12, s2_mlp_ratio=3), 'etc': {}},
    'volo_d2_224': {'parameter': dict(H=224, W=224,
                                      s1_num=6, s1_dim=256, s1_head=8, s1_mlp_ratio=3,
                                      s2_num=18, s2_dim=512, s2_head=16, s2_mlp_ratio=3), 'etc': {}},
    'volo_d3_224': {'parameter': dict(H=224, W=224,
                                      s1_num=8, s1_dim=256, s1_head=8, s1_mlp_ratio=3,
                                      s2_num=28, s2_dim=512, s2_head=16, s2_mlp_ratio=3), 'etc': {}},
    'volo_d4_224': {'parameter': dict(H=224, W=224,
                                      s1_num=8, s1_dim=384, s1_head=12, s1_mlp_ratio=3,
                                      s2_num=28, s2_dim=768, s2_head=16, s2_mlp_ratio=3), 'etc': {}},
    'volo_d5_224': {'parameter': dict(H=224, W=224, stem_hidden_dim=128,
                                      s1_num=12, s1_dim=384, s1_head=12, s1_mlp_ratio=4,
                                      s2_num=36, s2_dim=768, s2_head=16, s2_mlp_ratio=4), 'etc': {}},
}


@register_model
class VOLO(nn.Module):
    def __init__(self, num_classes, s1_num, s1_dim, s1_head, s1_mlp_ratio, s2_num, s2_dim, s2_head, s2_mlp_ratio,
                 H, W, K=3, padding=1, stride=2, stem_hidden_dim=64, use_token_label=True, drop_path_rate=0.0):
        super().__init__()
        self.patch_embedding1 = PatchEmbedding(3, s1_dim, H, W, 8, use_stem=True, hidden_dim=stem_hidden_dim)
        self.patch_embedding2 = PatchEmbedding(s1_dim, s2_dim, H // 8, W // 8, 2, add_pe=True)
        self.stage1 = nn.Sequential(
            *[Outlooker(s1_dim, s1_head, s1_mlp_ratio, H // 8, W // 8, K, padding, stride) for _ in range(s1_num)])
        self.stage2 = nn.Sequential(
            *[SelfAttention(s2_dim, s2_mlp_ratio, s2_head, drop_path_rate=drop_path_rate * (i / s2_num)) for i in
              range(s2_num)])
        self.cls = nn.Sequential(*[ClassAttention(s2_dim, s2_mlp_ratio, s2_head) for _ in range(2)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, s2_dim))
        self.norm = nn.LayerNorm(s2_dim)
        self.classifier = nn.Linear(s2_dim, num_classes)

        # To check if params & flops are matched with timm version
        self.use_token_label = use_token_label
        if self.use_token_label:
            self.aux_head = nn.Linear(s2_dim, num_classes)

    def forward(self, x):
        x = self.stage1(self.patch_embedding1(x))
        x = self.stage2(self.patch_embedding2(x))
        cls_token, x = self.cls((self.cls_token.expand(x.size(0), -1, -1), x))
        cls_token = self.norm(cls_token)
        out = self.classifier(cls_token)

        # To check if params & flops are matched with timm version
        if self.use_token_label:
            x_aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * x_aux.max(1)[0]

        return out


def volo_d1_224(**kwargs):
    return VOLO(num_classes=1000, H=224, W=224,
                s1_num=4, s1_dim=192, s1_head=6, s1_mlp_ratio=3,
                s2_num=14, s2_dim=384, s2_head=12, s2_mlp_ratio=3, **kwargs)


def volo_d2_224(**kwargs):  # change from d1: increase layer & dim & head
    return VOLO(num_classes=1000, H=224, W=224,
                s1_num=6, s1_dim=256, s1_head=8, s1_mlp_ratio=3,
                s2_num=18, s2_dim=512, s2_head=16, s2_mlp_ratio=3, **kwargs)


def volo_d3_224(**kwargs):  # change from d2: increase layer
    return VOLO(num_classes=1000, H=224, W=224,
                s1_num=8, s1_dim=256, s1_head=8, s1_mlp_ratio=3,
                s2_num=28, s2_dim=512, s2_head=16, s2_mlp_ratio=3, **kwargs)


def volo_d4_224(**kwargs):  # change from d3: increase dim & head
    return VOLO(num_classes=1000, H=224, W=224,
                s1_num=8, s1_dim=384, s1_head=12, s1_mlp_ratio=3,
                s2_num=28, s2_dim=768, s2_head=16, s2_mlp_ratio=3, **kwargs)


def volo_d5_224(**kwargs):  # change from d4: increase layer & mlp ratio
    """volo d5 @ 224

    modification from paper (copied from author code)
    - stem_hidden_dim = 128
    """
    return VOLO(num_classes=1000, H=224, W=224, stem_hidden_dim=128,
                s1_num=12, s1_dim=384, s1_head=12, s1_mlp_ratio=4,
                s2_num=36, s2_dim=768, s2_head=16, s2_mlp_ratio=4, **kwargs)
