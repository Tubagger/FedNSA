import torch
from torch import nn
import torch.nn.functional as F
from opacus.layers import DPLSTM
import torchvision.models as models
from torchvision.models import mobilenet_v2, resnet18, shufflenet_v2_x2_0, squeezenet1_1
__all__ = ['lenet5', 'squeezenet', 'mnistcnn','cifarcnn','alexnet']
class ConvGN(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_channels)  # 替换为GroupNorm

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.res1 = ConvGN(channels, channels, num_groups)
        self.res2 = ConvGN(channels, channels, num_groups)

    def forward(self, x):
        out = self.res2(self.res1(x))
        out = out + x  # identity shortcut connection
        return out

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.convgn = ConvGN(in_channels, out_channels, num_groups)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.convgn(x))

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.layer = Layer(in_channels, out_channels, num_groups)
        self.residual = Residual(out_channels, num_groups)

    def forward(self, x):
        return self.residual(self.layer(x))

class ResNet9(nn.Module):
    def __init__(self, input_size=32, num_classes=10, num_groups=8):
        super().__init__()
        self.prep = ConvGN(3, 2 * input_size, num_groups)
        self.layer1 = ResLayer(2 * input_size, 4 * input_size, num_groups)
        self.layer2 = Layer(4 * input_size, 8 * input_size, num_groups)
        self.layer3 = ResLayer(8 * input_size, 16 * input_size, num_groups)
        self.linear = nn.Linear(16 * input_size, num_classes, bias=False)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class MobileNetV2_GN(nn.Module):
    def __init__(self, num_classes=10, num_groups=2):
        """
        初始化 MobileNetV2 模型，并将 BatchNorm2d 替换为 GroupNorm。
        
        Args:
            num_classes (int): 分类的类别数。
            num_groups (int): GroupNorm 的组数。
        """
        super(MobileNetV2_GN, self).__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups

        # 加载原始 MobileNetV2 模型
        self.model = mobilenet_v2(weights=None, num_classes=num_classes)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self._replace_bn_with_gn(self.model, num_groups)

    def _replace_bn_with_gn(self, model, num_groups):
        """
        将模型中的所有 BatchNorm2d 替换为 GroupNorm。
        """
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                setattr(model, name, nn.GroupNorm(num_groups, num_channels))
            else:
                self._replace_bn_with_gn(module, num_groups)

    def forward(self, x):
        """
        前向传播函数。
        """
        return self.model(x)

class ResNet18_GN(nn.Module):
    def __init__(self, num_classes=10, num_groups=4, dropout_prob=0.5):
        """
        初始化 ResNet-18 模型，并将 BatchNorm2d 替换为 GroupNorm。

        Args:
            num_classes (int): 分类的类别数。
            num_groups (int): GroupNorm 的组数。
        """
        super(ResNet18_GN, self).__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups

        # 加载原始 ResNet-18 模型
        self.model = resnet18(weights=None, num_classes=num_classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # 添加 Dropout
            nn.Linear(num_features, num_classes)
        )
        self._replace_bn_with_gn(self.model, num_groups)

    def _replace_bn_with_gn(self, model, num_groups):
        """
        将模型中的所有 BatchNorm2d 替换为 GroupNorm。
        """
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                setattr(model, name, nn.GroupNorm(num_groups, num_channels))
            else:
                self._replace_bn_with_gn(module, num_groups)

    def forward(self, x):
        """
        前向传播函数。
        """
        return self.model(x)

class ShuffleNetV2_LN(nn.Module):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet-18 模型，并将 BatchNorm2d 替换为 LayerNorm。

        Args:
            num_classes (int): 分类的类别数。
        """
        super(ShuffleNetV2_LN, self).__init__()
        self.num_classes = num_classes

        # 加载原始 ResNet-18 模型
        self.model = shufflenet_v2_x2_0(num_classes=num_classes)
        self.model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 替换全连接层
        self._replace_bn_with_ln(self.model)

    def _replace_bn_with_ln(self, model):
        """
        将模型中的所有 BatchNorm2d 替换为 LayerNorm。
        """
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                # 替换为 LayerNorm，归一化整个通道维度和空间维度
                model._modules[name] = DynamicLayerNorm(num_channels)
            else:
                # 递归替换子模块
                self._replace_bn_with_ln(module)

    def forward(self, x):
        """
        前向传播函数。
        """
        return self.model(x)


class ResNet18_LN(nn.Module):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet-18 模型，并将 BatchNorm2d 替换为 LayerNorm。

        Args:
            num_classes (int): 分类的类别数。
        """
        super(ResNet18_LN, self).__init__()
        self.num_classes = num_classes

        # 加载原始 ResNet-18 模型
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 替换全连接层
        self._replace_bn_with_ln(self.model)

    def _replace_bn_with_ln(self, model):
        """
        将模型中的所有 BatchNorm2d 替换为 LayerNorm。
        """
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                # 替换为 LayerNorm，归一化整个通道维度和空间维度
                model._modules[name] = DynamicLayerNorm(num_channels)
            else:
                # 递归替换子模块
                self._replace_bn_with_ln(module)

    def forward(self, x):
        """
        前向传播函数。
        """
        return self.model(x)


class DynamicLayerNorm(nn.Module):
    """
    动态 LayerNorm，适配输入的形状。
    """
    def __init__(self, num_channels):
        super(DynamicLayerNorm, self).__init__()
        self.num_channels = num_channels

    def forward(self, x):
        # 获取输入形状 [N, C, H, W]
        normalized_shape = x.shape[1:]  # 从 [C, H, W] 开始归一化
        layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=True).to(x.device)
        return layer_norm(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.model = models.squeezenet1_1(pretrained=False)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.conv2_drop = nn.Dropout2d()
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=10, droprate=0):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=4, stride=4),
        )
        if droprate > 0.:
            self.fc = nn.Sequential(nn.Dropout(droprate),
                                    nn.Linear(256, num_classes))
        else:
            self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.max_pool2d(x,x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MnistCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(MnistCNN, self).__init__()

        # -------- Feature extractor --------
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)   # 28x28 → 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.pool = nn.MaxPool2d(2, 2)                            # /2

        # -------- Classifier --------
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 7, 7]

        x = x.view(x.size(0), -1)               # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CifarCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CifarCNN, self).__init__()

        # -------- Feature extractor --------
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 🔥 关键：自适应池化，彻底解决尺寸问题
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # -------- Classifier --------
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 输入: [B, 3, 224, 224] 或 [B, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.gap(x)                  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)        # [B, 128]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model

def lenet5(**kwargs):
    """
    Constructs a ResNet model.
    """
    return LeNet5(**kwargs)


def squeezenet(**kwargs):
    return SqueezeNet(**kwargs)


def mnistcnn(**kwargs):
    return MnistCNN(**kwargs)

def cifarcnn(**kwargs):
    return CifarCNN(**kwargs)

