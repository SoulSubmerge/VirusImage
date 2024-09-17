from models import resnet
import torch

input_tensor = torch.randn(1, 3, 224, 224)

def main():
    model = resnet.resnet18AndMultiHeadAttention()
    checkpoint = torch.load("./weights/pretrain/valid_best_0_1726467958244.pth", weights_only=False)
    ckp_keys = list(checkpoint['model_state_dict'])
    cur_keys = list(model.state_dict())
    model_sd = model.state_dict()
    for ckp_key in ckp_keys:
        model_sd[ckp_key] = checkpoint['model_state_dict'][ckp_key]

    model.load_state_dict(model_sd)
    # 模型推理（前向传播）
    output = model(input_tensor)

    # 输出模型的结构和输出结果的维度
    print("Model structure:\n", model)
    print("\nOutput shape:", output.shape)
    print(output)




if __name__ == '__main__':
    main()