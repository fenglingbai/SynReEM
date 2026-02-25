"""
三维数组带重叠子块连通域提取工具
支持处理大型三维数组，通过划窗方式提取连通域，避免跨块重复检测
"""

import numpy as np
from scipy.ndimage import label
from typing import List, Tuple, Dict
import uuid


def extract_connected_domains(volume: np.ndarray, 
                             block_size: Tuple[int, int, int] = (64, 64, 64)
                             ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    从三维数组中提取所有连通域（使用重叠划窗策略）
    
    Args:
        volume: 输入三维数组 (z, y, x)，值为0或1
        block_size: 子块尺寸 (z, y, x)，建议每个维度为2的倍数
        
    Returns:
        regions: 连通域掩码列表（裁剪到最小包围盒）
        coords: 每个连通域的全局起始坐标 (z, y, x)
        
    算法说明：
        1. 使用50%重叠的滑窗遍历整个体数据
        2. 对每个子块检测连通域
        3. 每个连通域分配唯一ID（基于中心体素坐标）
        4. 遍历完成后合并相同ID的连通域片段
        
    Example:
        >>> import numpy as np
        >>> volume = np.random.randint(0, 2, (100, 100, 100))
        >>> regions, coords = extract_connected_domains(volume, (32, 32, 32))
        >>> print(f"Found {len(regions)} connected domains")
    """
    # 步长为子块尺寸的一半，确保50%重叠
    step = tuple(b // 2 for b in block_size)
    
    # 为所有体素分配临时ID（用于合并跨块连通域）
    region_id_map = np.zeros(volume.shape, dtype=np.uint64)
    next_id = 1
    
    # 存储每个ID对应的体素坐标列表
    region_voxels: Dict[int, List[Tuple[int, int, int]]] = {}
    
    # 第一遍：遍历所有子块，为每个连通域分配ID
    for z_start in range(0, volume.shape[0], step[0]):
        for y_start in range(0, volume.shape[1], step[1]):
            for x_start in range(0, volume.shape[2], step[2]):
                # 提取子块
                z_end = min(z_start + block_size[0], volume.shape[0])
                y_end = min(y_start + block_size[1], volume.shape[1])
                x_end = min(x_start + block_size[2], volume.shape[2])
                
                block = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                block_id_map = region_id_map[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # 标记连通域
                labeled, num_regions = label(block, structure=np.ones((3, 3, 3)))
                
                for region_id in range(1, num_regions + 1):
                    region_mask = (labeled == region_id)
                    coords_in_block = np.where(region_mask)
                    
                    # 计算全局坐标
                    global_coords = [
                        (z_start + z, y_start + y, x_start + x)
                        for z, y, x in zip(*coords_in_block)
                    ]
                    
                    # 查找这个连通域的体素是否已有ID
                    existing_id = None
                    for gz, gy, gx in global_coords:
                        if region_id_map[gz, gy, gx] > 0:
                            existing_id = region_id_map[gz, gy, gx]
                            break
                    
                    if existing_id is not None:
                        # 使用已有ID
                        assigned_id = existing_id
                    else:
                        # 分配新ID
                        assigned_id = next_id
                        next_id += 1
                        region_voxels[assigned_id] = []
                    
                    # 记录所有体素到该ID
                    for gz, gy, gx in global_coords:
                        region_id_map[gz, gy, gx] = assigned_id
                        region_voxels[assigned_id].append((gz, gy, gx))
    
    # 第二遍：处理合并后的连通域
    results_mask = []
    results_coord = []
    
    for region_id, voxel_list in region_voxels.items():
        if len(voxel_list) == 0:
            continue
        
        # 转换为numpy数组
        voxels = np.array(voxel_list)
        
        # 计算包围盒
        z_min, z_max = voxels[:, 0].min(), voxels[:, 0].max()
        y_min, y_max = voxels[:, 1].min(), voxels[:, 1].max()
        x_min, x_max = voxels[:, 2].min(), voxels[:, 2].max()
        
        # 创建掩码
        mask_shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
        mask = np.zeros(mask_shape, dtype=np.uint8)
        
        # 填充掩码
        for gz, gy, gx in voxel_list:
            mask[gz - z_min, gy - y_min, gx - x_min] = 1
        
        results_mask.append(mask)
        results_coord.append((z_min, y_min, x_min))
    
    return results_mask, results_coord


def crop_region_to_bbox(region_mask: np.ndarray, 
                        global_offset: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    将连通域裁剪到最小包围盒
    
    Args:
        region_mask: 连通域掩码
        global_offset: 全局起始坐标
        
    Returns:
        cropped_mask: 裁剪后的掩码
        new_global_offset: 新的全局起始坐标
    """
    coords = np.argwhere(region_mask)
    if len(coords) == 0:
        return region_mask, global_offset
    
    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_min, x_max = coords[:, 2].min(), coords[:, 2].max()
    
    cropped = region_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    new_offset = (
        global_offset[0] + z_min,
        global_offset[1] + y_min,
        global_offset[2] + x_min
    )
    
    return cropped, new_offset


if __name__ == "__main__":
    # 测试用例1：简单立方体
    print("="*60)
    print("测试用例1：简单立方体")
    print("="*60)
    
    volume1 = np.zeros((100, 100, 100), dtype=np.uint8)
    
    # 添加几个测试连通域
    # 1. 内部小连通域 (10x10x10)
    volume1[20:30, 20:30, 20:30] = 1
    # 2. 大型立方体 (40x40x40)
    volume1[40:80, 40:80, 40:80] = 1
    # 3. 边界连通域 (10x10x10)
    volume1[0:10, 50:60, 50:60] = 1
    
    print("\n处理连通域...")
    block_size = (32, 32, 32)
    regions, coords = extract_connected_domains(volume1, block_size)
    
    print(f"\n找到 {len(regions)} 个连通域:")
    for i, (mask, coord) in enumerate(zip(regions, coords), 1):
        voxel_count = np.sum(mask)
        print(f"  {i}. 坐标: {coord}, 形状: {mask.shape}, 体素数: {voxel_count}")
    
    # 验证
    total_voxels = sum(np.sum(mask) for mask in regions)
    original_voxels = np.sum(volume1)
    print(f"\n验证:")
    print(f"  提取的连通域总像素数: {total_voxels}")
    print(f"  原始数组非零像素数: {original_voxels}")
    print(f"  {'✓' if total_voxels == original_voxels else '✗'} 验证{'通过' if total_voxels == original_voxels else '失败'}")
    
    # 测试用例2：多个小连通域
    print("\n" + "="*60)
    print("测试用例2：多个小连通域")
    print("="*60)
    
    np.random.seed(42)
    volume2 = np.zeros((50, 50, 50), dtype=np.uint8)
    # 添加10个随机小连通域
    for i in range(10):
        z, y, x = np.random.randint(5, 45, 3)
        volume2[z:z+5, y:y+5, x:x+5] = 1
    
    regions2, coords2 = extract_connected_domains(volume2, (16, 16, 16))
    
    print(f"\n找到 {len(regions2)} 个连通域:")
    for i, (mask, coord) in enumerate(zip(regions2, coords2), 1):
        voxel_count = np.sum(mask)
        print(f"  {i}. 坐标: {coord}, 体素数: {voxel_count}")
    
    total_voxels2 = sum(np.sum(mask) for mask in regions2)
    original_voxels2 = np.sum(volume2)
    print(f"\n验证: {'✓' if total_voxels2 == original_voxels2 else '✗'} "
          f"提取{total_voxels2} == 原始{original_voxels2}")
    
    # 测试用例3：跨块连通域
    print("\n" + "="*60)
    print("测试用例3：跨块连通域")
    print("="*60)
    
    volume3 = np.zeros((80, 80, 80), dtype=np.uint8)
    # 创建一个跨越多个子块的连通域
    volume3[15:65, 15:65, 15:65] = 1  # 50x50x50立方体
    
    regions3, coords3 = extract_connected_domains(volume3, (32, 32, 32))
    
    print(f"\n找到 {len(regions3)} 个连通域:")
    for i, (mask, coord) in enumerate(zip(regions3, coords3), 1):
        voxel_count = np.sum(mask)
        print(f"  {i}. 坐标: {coord}, 形状: {mask.shape}, 体素数: {voxel_count}")
    
    total_voxels3 = sum(np.sum(mask) for mask in regions3)
    original_voxels3 = np.sum(volume3)
    print(f"\n验证: {'✓' if total_voxels3 == original_voxels3 else '✗'} "
          f"提取{total_voxels3} == 原始{original_voxels3}")
