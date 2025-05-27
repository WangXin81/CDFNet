import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial

# from statsmodels.sandbox.regression.example_kernridge import stride

from models.base_block import *
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class Backbone(nn.Module):
    def __init__(self, patch_size=7, in_chans=3, num_classes=7, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.baseconv_01 = nn.Conv2d(32, 32, 3, 1, 1)
        self.baseconv_02 = nn.Conv2d(64, 64, 3, 1, 1)
        self.baseconv_03 = nn.Conv2d(128, 128, 3, 1, 1)
        self.baseconv_04 = nn.Conv2d(256, 256, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0]
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1]
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2]
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3]
                )
                for i in range(depths[3])
            ]
        )

        self.CT1 = MDM(32)
        self.CT2 = MDM(64)
        self.CT3 = MDM(128)
        self.CT4 = MDM(256)
        self.norm4 = norm_layer(embed_dims[3])
        self.cha_conv_1 = nn.Conv2d(256, 128,3, 1, 1)
        self.cha_conv_2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.cha_conv_3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.cha_conv_4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.cha_dif = nn.Conv2d(16, 1, 3, 1, 1)
        self.classifier1_1 = nn.Conv2d(256, 64, kernel_size=1)
        self.classifier1_2 = nn.Conv2d(128, 7, kernel_size=1)
        self.classifier2_1 = nn.Conv2d(256, 64, kernel_size=1)
        self.classifier2_2 = nn.Conv2d(128, 7, kernel_size=1)
        self.Class1 = IFFM()
        self.Class2 = IFFM()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # ----------------------------------#
        #   block1
        # ----------------------------------#
        x, H, W ,x0= self.patch_embed1.forward(x)
        x0 =  self.baseconv_01(x0)
        for i, blk in enumerate(self.block1):
            x = blk.forward(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x+x0)

        # ----------------------------------#
        #   block2
        # ----------------------------------#
        x, H, W,x0 = self.patch_embed2.forward(x)
        x0 = self.baseconv_02(x0)
        for i, blk in enumerate(self.block2):
            x = blk.forward(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x+x0)

        # ----------------------------------#
        #   block3
        # ----------------------------------#
        x, H, W ,x0= self.patch_embed3.forward(x)
        x0 = self.baseconv_03(x0)
        for i, blk in enumerate(self.block3):
            x = blk.forward(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x+x0)

        # ----------------------------------#
        #   block4
        # ----------------------------------#
        x, H, W ,x0= self.patch_embed4.forward(x)
        x0 = self.baseconv_04(x0)
        for i, blk in enumerate(self.block4):
            x = blk.forward(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x+x0)

        return outs

    def forward(self, x1 , x2):
        x1 = self.forward_features(x1)
        x2 = self.forward_features(x2)
        x_dif0 = self.CT4(x1[3],x2[3])
        x_dif0_0 = self.up(x_dif0)
        x_dif0_0 = self.cha_conv_1(x_dif0_0)
        x_dif1 = self.CT3(x1[2],x2[2],x_dif0_0)
        x_dif1_1 = self.up(x_dif1)
        x_dif1_1 = self.cha_conv_2(x_dif1_1)
        x_dif2 = self.CT2(x1[1], x2[1], x_dif1_1)
        x_dif2_2 = self.up(x_dif2)
        x_dif2_2 = self.cha_conv_3(x_dif2_2)
        x_dif3 = self.CT1(x1[0], x2[0], x_dif2_2)
        x_dif3_3 = self.up(x_dif3)
        x_dif3_3 = self.cha_conv_4(x_dif3_3)
        out_diff = self.up(x_dif3_3)
        out_diff = self.cha_dif(out_diff)
        x_class1 = self.Class1(x1[3],x_dif3,x_dif2,x_dif1,x_dif0)
        x_class2 = self.Class2(x2[3],x_dif3,x_dif2,x_dif1,x_dif0)
        return out_diff,x_class1,x_class2


class OverlapPatchEmbed(nn.Module):
    def __init__(self,patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x0 = self.proj(x)
        _, _, H, W = x0.shape
        x1 = x0.flatten(2).transpose(1, 2)
        x1 = self.norm(x1)
        return x1, H, W, x0

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # bs, 16384, 32 => bs, 16384, 32
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MDM(nn.Module):
    def __init__(self, inc):
        super(MDM, self).__init__()
        self.conv_jc0 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc0 = nn.BatchNorm2d(inc)
        self.conv_jc1_1 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc1_1 = nn.BatchNorm2d(inc)
        self.conv_jc1_2 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc1_2 = nn.BatchNorm2d(inc)
        self.conv_jc2 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc2 = nn.BatchNorm2d(inc)

        self.conv_jd = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jd = nn.BatchNorm2d(inc)
        self.conv_fusion = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_fusion = nn.BatchNorm2d(inc)

    def forward(self, feat1, feat2, pred=None):  ##pred相当于mask
        feat_jc = feat1 * feat2
        if pred is not None:
            feat_jc = feat_jc * pred
        feat_jc = F.relu(self.bn_jc0(self.conv_jc0(feat_jc)))
        feat_jc1 = F.relu(self.bn_jc1_1(self.conv_jc1_1(feat1 + feat_jc)))
        feat_jc2 = F.relu(self.bn_jc1_2(self.conv_jc1_2(feat2 + feat_jc)))
        feat_jc = F.relu(self.bn_jc2(self.conv_jc2(feat_jc1 + feat_jc2)))

        feat_jd = torch.abs(feat1 - feat2)
        if pred is not None:
            feat_jd = feat_jd * pred
        feat_jd = F.relu(self.bn_jd(self.conv_jd(feat_jd)))
        feat_fusion = F.relu(self.bn_fusion(self.conv_fusion(feat_jd + feat_jc)))
        return feat_fusion


class IFFM(nn.Module):  # 修正了类名拼写
    def __init__(self):
        super(IFFM, self).__init__()
        self.conv_cat1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv_cat2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv_cat3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv_cat4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 7, kernel_size=3, stride=1, padding=1)
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(7, 7, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x, diff1, diff2, diff3, diff4):
        x = torch.cat([x, diff4], dim=1)
        x = self.conv_cat1(x)
        x = self.decoder1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv0(x)
        x = self.bn1(x)

        x = torch.cat([x, diff3], dim=1)
        x = self.conv_cat2(x)
        x = self.decoder2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.bn2(x)

        x = torch.cat([x, diff2], dim=1)
        x = self.conv_cat3(x)
        x = self.decoder3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.bn3(x)

        x = torch.cat([x, diff1], dim=1)
        x = self.conv_cat4(x)
        x = self.decoder4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x_out = F.interpolate(self.conv3(x), scale_factor=2, mode='bilinear', align_corners=False)


        return x_out


if __name__ == "__main__":
    #a = torch.randn(1, 256, 16, 16)
    #diff1 = torch.randn(1, 32, 128, 128)
    #diff2 = torch.randn(1, 64, 64, 64)
    #diff3 = torch.randn(1, 128, 32, 32)
    #diff4 = torch.randn(1, 256, 16, 16)

    #model = CD_Decoder()
    #out = model(a, diff1, diff2, diff3, diff4)  # 使用模型实例调用forward方法
    #print(out.shape)

    a = torch.randn(1, 3, 512, 512)
    b = torch.randn(1, 3, 512, 512)
    a1 = torch.randn(1,64,128,128)
    a2 = torch.randn(1,64,128,128)
    pred = torch.randn(1,64,128,128)
    model2 = Backbone()
    Attention = MDM(64)
    out1 = Attention(a1,a2,pred)
    out2= model2(a,b)
    #print("out",x1.shape,x2.shape,x3.shape,x4.shape)
    print(out2[0].shape,out2[1].shape,out2[2].shape)
    #print(out2.shape)