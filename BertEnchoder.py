import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
class BERTmbeadings(nn.Module):
    def __init__(self, vocab_size, embedDim, numPosEmbeading, numSegEmbeading, padIdx = None):
        super().__init__()
        self.Embead = nn.Embedding(vocab_size, embedDim, padIdx)
        self.PosEmbead = nn.Embedding(numPosEmbeading, embedDim)
        self.SegEmbeading = nn.Embedding(numSegEmbeading, embedDim)
    def forward(self, tokens, seg = None, pos = None):
        if pos == None :
            pos = torch.arange(tokens.shape[1], device=tokens.device)
            pos = pos.unsqueeze(0)
            pos = pos.expand(tokens.shape[0], -1)
        if seg == None:
            seg = torch.zeros_like(tokens)

        res = self.Embead(tokens) + self.PosEmbead(pos) + self.SegEmbeading(seg)
        return res

class EncoderBLock(nn.Module):
    def __init__(self, embedDim, num_heads, dropout = 0.1):
        super().__init__()
        self.embedDim = int(int(embedDim / num_heads) * num_heads)
        self.Attantion = nn.MultiheadAttention(self.embedDim, num_heads, dropout= dropout, batch_first= True)
        self.MLP = nn.Sequential(
            nn.Linear(self.embedDim, self.embedDim * 4),
            nn.GELU(),
            nn.Linear(self.embedDim * 4, self.embedDim)
        )
        self.Norm1 = nn.LayerNorm(self.embedDim)
        self.Norm2 = nn.LayerNorm(self.embedDim)
        self.Drop = nn.Dropout(dropout,inplace=True)
    def forward(self, inp, mask = None):
        a_o, _ = self.Attantion.forward(inp, inp , inp, key_padding_mask=mask)
        l1 = self.Norm1(inp + self.Drop(a_o))
        m_o = self.MLP(l1)
        op = self.Norm2(m_o + l1)
        return op

class MaskedLangModeling(nn.Module):
    def __init__(self,vocabSize : int, embedDim : int):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, vocabSize)
        )
    def forward(self, seq):
        return self.MLP(seq)

class NextSentPred(nn.Module):
    def __init__(self, embedDim : int, idx = 0):
        super().__init__()
        self.idx = idx
        self.MLP = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, 1)
        )
    def forward(self, seq):
        return self.MLP(seq[:, self.idx, :]).view(-1)


class EncoderOnly(nn.Module):
    def __init__(self, vocabSize, embedDim = 192, numHeads = 12, numLayers = 12, numPosEmbeading=128, numSegEmbeading = 2, padIdx = 0):
        super().__init__()
        embedDim = int(int(embedDim / numHeads) * numHeads)
        self.cfgDict = {"vocabSize" : vocabSize,
                        "embedDim" : embedDim,
                        "numHeads" : numHeads,
                        'numLayers' : numLayers,
                        "numPosEmbeading" : numPosEmbeading,
                        "numSegEmbeading" : numSegEmbeading,
                        "padIdx" : padIdx
                        }
        self.Embead = BERTmbeadings(vocabSize, embedDim, numPosEmbeading, numSegEmbeading, padIdx)
        self.Blocks = nn.ModuleList()
        for i in range(numLayers):
            self.Blocks.append(EncoderBLock(embedDim, numHeads))

        self.MLM = MaskedLangModeling(vocabSize, embedDim)
        self.NSP = NextSentPred(embedDim)
    def forward(self, tokens, seg = None, mask = None, pos = None):
        x = self.Embead.forward(tokens, seg, pos)
        for Block in self.Blocks:
            x = Block.forward(x, mask)

        MLMOut = self.MLM(x)
        NSPOut = self.NSP(x)
        return MLMOut, NSPOut
