import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from MultiheadAttention import Attention

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        self.encoding = self._get_position_embedding(seq_length, embed_dim)

    @staticmethod
    def _get_position_embedding(seq_length, embed_dim):
        position_embedding = torch.zeros(seq_length, embed_dim)
        for pos in range(seq_length):
            for i in range(0, embed_dim, 2):
                position_embedding[pos, i] = torch.sin(
                    torch.tensor(pos) / (10000 ** (torch.tensor(i) / embed_dim)))
                if i + 1 < embed_dim:
                    position_embedding[pos, i + 1] = torch.cos(
                        torch.tensor(pos) / (10000 ** (torch.tensor(i) / embed_dim)))
        return position_embedding.unsqueeze(0)

    def forward(self, x):

        if x.device != self.encoding.device:
            self.encoding = self.encoding.to(x.device)

        return x + self.encoding


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        #self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention =Attention(embed_dim, embed_dim,num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        return x

    def get_attention(self):
<<<<<<< HEAD
        return self.attention.get_attention()
=======
        return self.attention.get_attention()  # 返回当前块的注意力权重
>>>>>>> 711b3a78cb0c68e5034894d8e2e704545a646e15

class VisionTransformer(nn.Module):
    def __init__(self, img_size=50, patch_size=10, num_classes=64, embed_dim=48, num_heads=4, ff_dim=256, num_layers=3, dropout_rate=0.):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embedding = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = PositionalEncoding(embed_dim, self.num_patches)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)
        # Patch embedding
        x = self.patch_embedding(x)  # [B, E, H' , W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, E]
        B, num_patches, E=x.shape
        x = self.positional_encoding(x).reshape(B,1, num_patches, E)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x=x.reshape(B, num_patches, E)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_attention_maps(self):
        attention_maps = [block.get_attention() for block in self.transformer_blocks]
        return attention_maps
        # Return :  the attention weights of all blocks