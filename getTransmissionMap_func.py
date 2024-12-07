import torch
import torch.nn.functional as F

# 假设getTransmission和stretching函数已经适配为PyTorch版本
from getTransmissionMap import getTransmission  # 确保这个函数已经转换为PyTorch版本
from global_histogram_stretching import stretching  # 确保这个函数已经转换为PyTorch版本

def getMinChannel(img, AtomsphericLight):
    """
    计算每个像素点在三个通道上的最小值归一化结果。
    
    :param img: 输入图像，形状为 [bs, C, H, W]
    :param AtomsphericLight: 大气光，形状为 [bs, 3]
    :return: 归一化的最小通道图像，形状为 [bs, 1, H, W]
    """
    # 归一化每个通道
    imgNormalization = img / AtomsphericLight[:, :, None, None]
    
    # 找到每个像素点的最小值
    imgGrayNormalization, _ = torch.min(imgNormalization, dim=1, keepdim=True)
    
    return imgGrayNormalization

def getTransmission(img, AtomsphericLight, blockSize):
    """
    计算传输图。
    
    :param img: 输入图像，形状为 [bs, C, H, W]
    :param AtomsphericLight: 大气光，形状为 [bs, 3]
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
    max_AtomsphericLight, _ = torch.max(AtomsphericLight, dim=1, keepdim=True)
    max_AtomsphericLight = max_AtomsphericLight[:, :, None, None]  # 扩展维度以匹配imgDark
    transmission = (1 - imgDark) / (1 - 0.1 / max_AtomsphericLight)
    
    # 限制传输图的范围
    transmission = torch.clamp(transmission, 0.1, 0.9)
    
    return transmission

def get_transmission(img):
    """
    计算传输图并进行拉伸。
    
    :param img: 输入图像，形状为 [bs, C, H, W]
    :return: 拉伸后的传输图，形状为 [bs, 1, H, W]
    """
    bs, C, H, W = img.shape
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 1e-3  # 引导滤波时epsilon的值
    blockSize = 9
    Nrer = [0.95, 0.93, 0.85]

    # 计算大气光
    AtomsphericLight = torch.zeros((bs, 3), device=img.device)
    AtomsphericLight[:, 0] = (1.13 * torch.mean(img[:, 0, :, :], dim=(1, 2))) + 1.11 * torch.std(img[:, 0, :, :], dim=(1, 2)) - 25.6
    AtomsphericLight[:, 1] = (1.13 * torch.mean(img[:, 1, :, :], dim=(1, 2))) + 1.11 * torch.std(img[:, 1, :, :], dim=(1, 2)) - 25.6
    # 计算第三个通道的中位数
    median_values, _ = torch.median(img[:, 2, :, :], dim=1, keepdim=True)
    median_values, _ = torch.median(median_values, dim=2, keepdim=True)
    median_values = median_values.squeeze(1).squeeze(1)  # 去掉多余的维度
    AtomsphericLight[:, 2] = 140 / (1 + 14.4 * torch.exp(-0.034 * median_values))
    AtomsphericLight = torch.clamp(AtomsphericLight, 10, 245)

    #print('AtomsphericLight', AtomsphericLight)

    # 获取传输图
    transmissionR = getTransmission(img, AtomsphericLight, blockSize)
    transmissionR_new = transmissionR
    #print('1', transmissionR_new.shape)

    # 对传输图进行拉伸
    transmissionR_Stretched = stretching(transmissionR_new)
    #print('1', transmissionR_Stretched.shape)

    return transmissionR_Stretched

if __name__ == '__main__':
    # 示例用法
    # 假设我们有一个批次的图像
# =============================================================================
#     batch_size = 2
#     C, H, W = 3, 256, 256
#     img = torch.randn(batch_size, C, H, W, dtype=torch.float32)  # 随机生成一批图像
# =============================================================================
    import cv2
    import os
    import numpy as np
    from os.path import join as join
    
    img1 = cv2.imread(r"C:\Users\lfc1995\Desktop\test_images\InputImages\test4.png")
    img2 = cv2.imread(r"C:\Users\lfc1995\Desktop\test_images\InputImages\test5.png")
    output_dir = r'C:\Users\lfc1995\Desktop\1'
    os.makedirs(output_dir, exist_ok=True)

    def img2tensor(img):
        # 转换颜色通道顺序以匹配PyTorch (H, W, C) -> (C, H, W)
        img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        # 将图像转换为PyTorch张量
        img_tensor = torch.from_numpy(img).float()
        return img_tensor
    
    img = torch.stack([img2tensor(img1), img2tensor(img2)]) # [2, 3, h, w]

    # 调用get_transmission函数
    transmissionR_Stretched = get_transmission(img)
    transmissionR_Stretched = transmissionR_Stretched.numpy() * 255
    for n, tm in enumerate(transmissionR_Stretched):
        output_path = join(output_dir, f"{n}_transmission.png")
        cv2.imwrite(output_path, tm[0].astype(np.uint8))
    #print(transmissionR_Stretched.shape)  # 输出应该是 [bs, 1, H, W]