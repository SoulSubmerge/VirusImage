import torch



class SelfAttention(torch.nn.Module):
    def __init__(self, in_planes):
        super(SelfAttention, self).__init__()
        self.query_conv = torch.nn.Conv2d(in_planes, in_planes // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_planes, in_planes // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Compute query, key, and value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C x N
        value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N

        # Attention map
        attention = torch.bmm(query, key)  # B x N x N
        attention = self.softmax(attention)  # B x N x N

        # Compute weighted value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        # Apply a learnable scaling factor
        out = self.gamma * out + x

        return out


class BasicAttentionBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicAttentionBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample

        # 插入自注意力模块
        self.attention = SelfAttention(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 加入自注意力层
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, in_planes, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert in_planes % num_heads == 0, "The number of input planes must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_planes // num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        # Linear layers to project inputs to query, key, and value
        self.query_conv = torch.nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_planes, in_planes, kernel_size=1)
        
        # Linear layer to combine heads after attention
        self.out_conv = torch.nn.Conv2d(in_planes, in_planes, kernel_size=1)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Generate query, key, and value for multi-heads
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, width * height)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, width * height)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, width * height)
        
        # Transpose for correct matrix multiplication: (B, num_heads, head_dim, N) -> (B, num_heads, N, head_dim)
        query = query.permute(0, 1, 3, 2)
        key = key.permute(0, 1, 2, 3)  # Keep key as (B, num_heads, head_dim, N)
        
        # Attention scores: Q * K^T
        attention = torch.matmul(query, key) * self.scale  # (B, num_heads, N, N)
        attention = self.softmax(attention)  # Softmax along the last dimension (N, N)
        
        # Compute weighted value: attention * V
        value = value.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, N) -> (B, num_heads, N, head_dim)
        out = torch.matmul(attention, value)  # (B, num_heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, N) -> (B, num_heads, N, head_dim)
        
        # Combine heads: concatenate and project
        out = out.view(batch_size, C, width, height)
        out = self.out_conv(out)
        
        # Apply a learnable scaling factor
        out = self.gamma * out + x

        return out


class BasicMultiHeadAttentionBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, num_heads=8):
        super(BasicMultiHeadAttentionBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample

        # 插入多头注意力机制
        self.attention = MultiHeadAttention(planes, num_heads=num_heads)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 加入多头注意力层
        outAtten = self.attention(out)
        out = out + outAtten
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out



class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out