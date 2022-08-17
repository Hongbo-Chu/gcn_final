import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as encoderblock
from timm.models.layers import PatchEmbed
"""
use timm.vit_base_patch8_224

architecture of vit
PatchEmbed
Droupout

encoder block * 12

layerNorm
Identity
Linear
"""
class vit(nn.Module):
    def __init__(self, block_num, pretrain = None, embed_dim=768):
        """
        params:
        block_num: num of the vit block, ranging from 0-11
        pretrain: the path of the .pth
        """
        super(vit, self).__init__()
        
        # if pretrain is not None:#预训练模型采用无监督自行训练
        #     self.basemodel.load_state_dict(torch.load(self.pretrain))
        
        # #开始精简模型
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=16, in_chans=3, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=0)
        self.blocks = nn.Sequential(*[
                encoderblock( 
                dim=embed_dim, num_heads=3, mlp_ratio=4, qkv_bias=False, drop=0,
                attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm, act_layer=nn.GELU)
                    for _ in range(1)])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_tokens = 2 
        self.num_patches = 195
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.aap = torch.nn.AdaptiveAvgPool2d(224)
    def forward(self, x):# TODO distillation token是啥
        x = self.aap(x) 
        # print(f"1.{id(x)},{x.device}")
        x = self.patch_embed(x)
        # print(f"2.{id(x)},{x.device}")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        return x
        
def buildvit():
    return vit(1, pretrain=None)



if __name__ =='__main__':
    a = vit(1,pretrain=None)
    b = a(torch.randn(1,3,224,224))
    print(b.size())
        #网络参数数量
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    kk = get_parameter_number(a)
    print(kk)
    for name in a.state_dict():
        print(name)