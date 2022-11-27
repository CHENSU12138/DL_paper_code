
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def image2emb_conv(image,kernel,stride):
    '''基于2维卷积来实现patch embedding，embedding的维度就是卷积的输出通道数'''
    conv_output = F.conv2d(image,kernel,stride=stride)
    bs,oc,oh,ow = conv_output.shape #batch_size output_channel,height,width
    patch_embedding = conv_output.reshape((bs.oc,oh*ow)).transpose(-1,-2) # num_patch = height*width [bs,num+patch,model_dim_C]
    return patch_embedding


class MultiHeadSelfAttention(nn.module):
    '''基于输入x惊醒三个映射分别得到qkv，将qkv拆分成多头的形式'''
    def __init__(self,model_dim,num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.proj_linear_layer = nn.Linear(model_dim,3*model_dim) # 拆分成qkv
        self.final_linear_layer = nn.Linear(model_dim,model_dim)

    def forward(self,input,additive_mask = None):
        bs,seqlen,model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head

        proj_output = self.proj_linear_layer(input) # [bs,seqlen,3*model_dim]
        q,k,v = proj_output.chunk(3,dim = 1) # 3*...

        q = q.reshape(bs,seqlen,num_head,head_dim).transpose(1,2) # bs,num_head,seqlen,head_dim
        q = q.reshape(bs*num_head,seqlen,head_dim)

        k = k.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2)  # bs,num_head,seqlen,head_dim
        k = k.reshape(bs * num_head, seqlen, head_dim)

        v = v.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2)  # bs,num_head,seqlen,head_dim
        v = v.reshape(bs * num_head, seqlen, head_dim)

        if additive_mask is None:
            attn_prob = F.softmax(torch.bmm(q,k.transpose(-2,-1))/math.sqrt(head_dim),dim = -1)

        else:
            additive_mask = additive_mask.tile((num_head,1,1))
            attn_prob = F.softmax(torch.bmm(q,k.transpose(-2,-1))/math.sqrt(head_dim)+additive_mask,dim = -1)

        output = torch.bmm(attn_prob,v) # [bs*num_head,seqlen,head_dim]
        output = output.reshape(bs,num_head,seqlen,head_dim).transpose(1,2) #[ns,seqlen,num_head,head_Dim]
        output = output.reshape(bs,seqlen,model_dim)
        output = self.final_linear_layer(output)
        return  attn_prob, output


def window_multi_head_self_attention(patch_embedding,mhsa,window_size = 4,num_head = 2):
    num_patch_in_windwos = window_size * window_size
    bs,num_patch,patch_depth = patch_embedding.shape





