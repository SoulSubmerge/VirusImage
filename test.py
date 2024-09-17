import torch
import math


class TokenEmbedding(torch.nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)



def main():
    encoding = torch.zeros(5,6)
    pos = torch.arange(0,5).unsqueeze(dim=1)
    encoding[:,0::2] = torch.sin(pos*torch.exp(torch.arange(0., 6, 2) * (-math.log(10000.0) / 6)))
    #encoding[:,1::2] = torch.cos(pos*torch.exp(torch.arange(0., 6, 2) * (-math.log(10000.0) / 6)))
    print(encoding)
    wq = torch.nn.Linear(6,2)
    xx = wq(encoding)
    print(xx)


if __name__ == '__main__':

    # 加载整个模型
    checkpoint = torch.load("./models/pretrain/valid_best_0_1726467958244.pth")
    ckp_keys = list(checkpoint['model_state_dict'])
    print(ckp_keys)
    ckp_keys = ckp_keys[:120]
    for ckp_key in ckp_keys:
        print(ckp_key)