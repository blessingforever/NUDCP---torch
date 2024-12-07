import torch
import torch.nn.functional as F

def getMinChannel(img, AtomsphericLight):
    """
    计算每个像素点在三个通道上的最小值归一化结果。
    
    :param img: 输入图像，形状为 [bs, C, H, W]
    :param AtomsphericLight: 大气光，形状为 [bs, 3] 或 [3]
    :return: 归一化的最小通道图像，形状为 [bs, 1, H, W]
    """
    # 如果AtomsphericLight是单个样本的大气光，则扩展到与img相同的batch大小
    if len(AtomsphericLight.shape) == 1:
        AtomsphericLight = AtomsphericLight.unsqueeze(0).repeat(img.size(0), 1)
    
    # 归一化每个通道
    imgNormalization = img / AtomsphericLight[:, :, None, None]
    
    # 找到每个像素点的最小值
    imgGrayNormalization, _ = torch.min(imgNormalization, dim=1, keepdim=True)
    
    return imgGrayNormalization

def getTransmission(img, AtomsphericLight, blockSize):
    """
    计算传输图。
    
    :param img: 输入图像，形状为 [bs, C, H, W]
    :param AtomsphericLight: 大气光，形状为 [bs, 3] 或 [3]
    :param blockSize: 块大小
    :return: 传输图，形状为 [bs, 1, H, W]
    """
    # 获取最小通道图像
    imgGrayNormalization = getMinChannel(img, AtomsphericLight)
    
    # 将大气光归一化到 [0, 1]
    AtomsphericLight = AtomsphericLight / 255.0
    
    # 计算需要填充的尺寸
    addSize = (blockSize - 1) // 2
    padding = (addSize, addSize, addSize, addSize)
    
    # 对图像进行填充
    imgMiddle = F.pad(imgGrayNormalization, padding, mode='constant', value=1.0)
    
    # 使用最大池化来找到局部最小值
    imgDark = -F.max_pool2d(-imgMiddle, kernel_size=blockSize, stride=1)
    
    # 计算传输图
    transmission = (1 - imgDark) / (1 - 0.1 / torch.max(AtomsphericLight, dim=1, keepdim=True)[0])
    
    # 限制传输图的范围
    transmission = torch.clamp(transmission, 0.1, 0.9)
    
    return transmission

if __name__ == '__main__':
    # 示例用法
    # 假设我们有一个批次的图像和相应的大气光
    batch_size = 2
    C, H, W = 3, 256, 256
    img = torch.randn(batch_size, C, H, W)  # 随机生成一批图像
    AtomsphericLight = torch.tensor([[100, 100, 100], [100, 100, 100]], dtype=torch.float32)  # 两个样本的大气光
    blockSize = 9
    
    transmission = getTransmission(img, AtomsphericLight, blockSize)
    print(transmission.shape)  # 输出应该是 [bs, 1, H, W]