import torch

def global_stretching_depth(img_L):
    # 计算 I_min 和 I_max
    height, width = img_L.shape
    num_pixels = height * width
    k = int(num_pixels / 1000)
    
    # 找到最小的 k 个值
    I_min, _ = torch.kthvalue(img_L.view(-1), k=k)
    
    # 找到最大的 k 个值
    I_max, _ = torch.topk(img_L.view(-1), k=k, largest=True, sorted=True)
    I_max = I_max[-1]  # 取最大的 k 个值中的最后一个值
    
    # 全局直方图拉伸
    # 首先将 img_L 夹在 I_min 和 I_max 之间
    clamped_img_L = torch.clamp(img_L, min=I_min, max=I_max)
    
    # 然后进行线性映射
    stretched_img_L = (clamped_img_L - I_min) * ((0.9 - 0.2) / (I_max - I_min)) + 0.2

    return stretched_img_L

def stretching(sceneRadiance):
    # 假设 sceneRadiance 是一个形状为 [bs, 1, H, W] 的张量
    batch_size, channels, height, width = sceneRadiance.shape
    
    # 创建一个空的张量来存储结果
    stretched_radiance = torch.zeros_like(sceneRadiance)
    
    # 对 batch 内的每个图像分别进行计算
    for i in range(batch_size):
        # 获取单个图像
        img_L = sceneRadiance[i, 0, :, :]
        
        # 对单个图像进行全局拉伸
        stretched_img_L = global_stretching_depth(img_L)
        
        # 将结果存入输出张量
        stretched_radiance[i, 0, :, :] = stretched_img_L

    #print('stretched_radiance shape after stretching:', stretched_radiance.shape)  # 调试信息
    return stretched_radiance

if __name__ == '__main__':
    # 示例用法
    # 假设我们有一个批次的图像
    batch_size = 2
    C, H, W = 1, 256, 256
    img = torch.randn(batch_size, C, H, W, dtype=torch.float32)  # 随机生成一批图像
    
    # 调用 get_transmission 函数
    transmissionR_Stretched = stretching(img)
    print(transmissionR_Stretched.shape)  # 输出应该是 [bs, 1, H, W]