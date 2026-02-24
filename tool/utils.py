import os
from os.path import join
import numpy 
import timeit
from skimage import io, measure, morphology
import numpy as np
from scipy.ndimage import label, zoom, binary_dilation, binary_fill_holes, binary_opening, binary_closing, binary_erosion, distance_transform_edt
from tqdm import tqdm
from skimage.segmentation import watershed
import cv2

def extract_largest_element(binary_image):
    # Label connected components
    labeled_image = measure.label(binary_image)
    # Calculate properties of connected components
    properties = measure.regionprops(labeled_image)
    # Find the largest connected component
    largest_component = None
    max_area = 0
    max_id = 0
    current_id = 0
    for prop in properties:
        current_id = current_id + 1
        if prop.area > max_area:
            max_area = prop.area
            largest_component = prop
            max_id = current_id
    
    if largest_component is None:
        return None, None
    largest_component_bbox = largest_component.bbox
    largest_component_image = labeled_image[largest_component_bbox[0]:largest_component_bbox[3], 
                                            largest_component_bbox[1]:largest_component_bbox[4], 
                                            largest_component_bbox[2]:largest_component_bbox[5]] == max_id
    return largest_component_bbox, largest_component_image

def multi_step_erosion_until_multiple_components(array, erosion_kernel, max_steps=3):
    array = array.astype(bool)
    for step in range(max_steps):
        eroded_array = binary_erosion(array, erosion_kernel)
        _, num_features = label(eroded_array)

        if num_features > 1:
            # return eroded_array
            return array, max_steps
        array = eroded_array
    return array, max_steps

def multi_step_dilation_until_single_component(array, dilation_kernel, max_steps=10):
    array = array.astype(bool)
    for step in range(max_steps):
        dilated_array = binary_dilation(array, dilation_kernel)
        # 标记连通域
        _, num_features = label(dilated_array)
        # 检查连通域的数量
        if num_features == 1:
            return dilated_array, step+1
        # 更新数组为膨胀后的数组
        array = dilated_array
    return array, step+1

def generate_ellipsoid_structuring_element(a, b, c, resolution=(1, 1, 1)):  
    # 根据resolution生成网格坐标  
    x = np.linspace(-a, a, int(resolution[0] * a) + 1)  
    y = np.linspace(-b, b, int(resolution[1] * b) + 1)  
    z = np.linspace(-c, c, int(resolution[2] * c) + 1)  
      
    # 生成网格  
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
      
    # 计算每个点到椭球中心的距离的平方  
    distances_squared = (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2  
      
    # 生成结构元素，内部为True，外部为False  
    se = distances_squared <= 1  
      
    return se 

def extract_2d_serial_edge(image_stack, edge_kernel=np.ones(shape=(1, 5, 5))):
    padded_image_stack = np.pad(image_stack, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    eroded_array = binary_erosion(padded_image_stack, edge_kernel).astype(np.uint8)
    # eroded_index = np.where(eroded_array > 0)
    # if len(eroded_index[0]) != 0:
    #     eroded_array[eroded_index[0].min(), :, :] = 0
    #     eroded_array[eroded_index[0].max(), :, :] = 0
    edge_stack = (padded_image_stack > eroded_array).astype(np.uint8)
    out_stack = edge_stack * 2 + eroded_array * 3
    return out_stack[1:-1, 1:-1, 1:-1]

def extract_3d_serial_edge(image_stack, edge_kernel):
    # generate_ellipsoid_structuring_element(2, 4, 4, resolution=(1, 1, 1))
    padded_image_stack = np.pad(image_stack, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    eroded_array = binary_erosion(padded_image_stack, edge_kernel).astype(np.uint8)
    edge_stack = (padded_image_stack > eroded_array).astype(np.uint8)
    out_stack = edge_stack * 2 + eroded_array * 3
    return out_stack[1:-1, 1:-1, 1:-1]

def skeletonize_2d_in_3d(image_stack):
    bin_image_stack = image_stack > 0
    unique_label_index = np.where(bin_image_stack)
    # z1, z2, y1, y2, x1, x2
    z1, z2, y1, y2, x1, x2 = [unique_label_index[0].min(), unique_label_index[0].max() + 1, 
                                unique_label_index[1].min(), unique_label_index[1].max() + 1, 
                                unique_label_index[2].min(), unique_label_index[2].max() + 1, ]
    
    part_stack = bin_image_stack[z1:z2, y1:y2, x1:x2].copy()
    out_stack = []
    for layer in range(part_stack.shape[0]):
        out_stack.append(morphology.skeletonize(part_stack[layer]))
    out_stack = np.array(out_stack)
    bin_image_stack[z1:z2, y1:y2, x1:x2] = out_stack
    return bin_image_stack

def get_single_region(bin_unique_stack, element_op, max_dilations, save_stack, bbox=None):
    ori_shape = bin_unique_stack.shape
    # 1.compute roi, need to compute big pad
    if bbox is not None:
        z1, z2, y1, y2, x1, x2 = bbox
        pad_z = element_op.shape[0] // 2 * max_dilations
        pad_y = element_op.shape[1] // 2 * max_dilations
        pad_x = element_op.shape[2] // 2 * max_dilations
        z1_use = z1 - pad_z if z1 > pad_z else 0
        y1_use = y1 - pad_y if y1 > pad_y else 0
        x1_use = x1 - pad_x if x1 > pad_x else 0
        z2_use = z2 + pad_z if z2 + pad_z < ori_shape[0] else ori_shape[0]
        y2_use = y2 + pad_y if y2 + pad_y < ori_shape[1] else ori_shape[1]
        x2_use = x2 + pad_x if x2 + pad_x < ori_shape[2] else ori_shape[2]
    else:
        z1, z2, y1, y2, x1, x2 = [0, ori_shape[0], 0, ori_shape[1], 0, ori_shape[2]]
        z1_use, z2_use, y1_use, y2_use, x1_use, x2_use = [0, ori_shape[0], 0, ori_shape[1], 0, ori_shape[2]]
    bin_roi_unique_stack = bin_unique_stack[z1_use:z2_use, y1_use:y2_use, x1_use:x2_use]
    roi_array, step_dilation = multi_step_dilation_until_single_component(bin_roi_unique_stack, dilation_kernel=element_op, 
                                                                            max_steps=max_dilations)
    roi_array_save = roi_array * (save_stack[z1_use:z2_use, y1_use:y2_use, x1_use:x2_use] == 0)
    save_stack[z1_use:z2_use, y1_use:y2_use, x1_use:x2_use][roi_array_save > 0] = 1

    # 2.compute edge and mask
    part_stack = bin_unique_stack[z1:z2, y1:y2, x1:x2].copy()
    part_stack = extract_2d_serial_edge(part_stack.astype(np.uint8), 
                                        edge_kernel=np.ones(shape=(1, 5, 5)))
    save_stack[z1:z2, y1:y2, x1:x2][part_stack>0] = part_stack[part_stack>0]

    # 3.core
    core_array = morphology.skeletonize_3d(roi_array)

    core_array = (core_array + skeletonize_2d_in_3d(bin_roi_unique_stack) > 0)

    core_array = binary_dilation(core_array, np.ones(shape=(1, 3, 3), dtype=np.uint8))

    core_array = core_array[z1-z1_use:z2-z1_use, y1-y1_use:y2-y1_use, x1-x1_use:x2-x1_use] * (part_stack==3)
    core_bbox, core_array = extract_largest_element(core_array)
    save_stack[z1 + core_bbox[0]:z1 + core_bbox[3], 
                y1 + core_bbox[1]:y1 + core_bbox[4], 
                x1 + core_bbox[2]:x1 + core_bbox[5], ][core_array>0] = core_array[core_array>0] * 4
    return save_stack


def structual_encoding(image_stack, structual_size=1, ani_scale=4):
    element_op = generate_ellipsoid_structuring_element(structual_size * 2, 
                                                structual_size * ani_scale * 2, 
                                                structual_size * ani_scale * 2, resolution=(1, 1, 1))  # 这里使用更精细的分辨率  
    

    padded_image_stack = np.pad(image_stack, pad_width=((structual_size, structual_size), 
                                                        (structual_size * ani_scale, structual_size * ani_scale), 
                                                        (structual_size * ani_scale, structual_size * ani_scale)), mode='edge')
    save_stack = np.zeros(shape=padded_image_stack.shape, dtype=np.uint8)
    unique_elements = np.unique(padded_image_stack)
    for unique_label in tqdm(unique_elements):
        # io.imsave(join(r'D:\Pycharm\datasets\AC3AC4_synapse', 'AC4Labels_encode.tif'), save_stack)
        if unique_label != 0:
            bin_unique_stack = padded_image_stack == unique_label
            unique_label_index = np.where(bin_unique_stack)
            # z1, z2, y1, y2, x1, x2
            z1, z2, y1, y2, x1, x2 = [unique_label_index[0].min(), unique_label_index[0].max() + 1, 
                                      unique_label_index[1].min(), unique_label_index[1].max() + 1, 
                                      unique_label_index[2].min(), unique_label_index[2].max() + 1, ]
            # uint8_unique_label_stack = bin_unique_stack.astype(np.uint8)
            # roi, edge, mask, core
            save_stack = get_single_region(bin_unique_stack, element_op=element_op, max_dilations=3, save_stack=save_stack, bbox=[z1, z2, y1, y2, x1, x2])

    return save_stack[structual_size:-structual_size, 
                      structual_size * ani_scale:-structual_size * ani_scale,
                      structual_size * ani_scale:-structual_size * ani_scale]



def structual_decoding(semantic_stack, aemc_stack, element_op, y_ture=None, graph_thre=1200, coonnect_thre=200, ani_scale=3):
    print('graph_thre ', graph_thre, 'coonnect_thre ', coonnect_thre, 'ani_scale ', ani_scale)
    padded_semantic_stack = np.pad(semantic_stack, pad_width=(((element_op.shape[0]-1) // 2, (element_op.shape[0]-1) // 2), 
                                                    ((element_op.shape[1]-1) // 2, (element_op.shape[1]-1) // 2), 
                                                    ((element_op.shape[2]-1) // 2,(element_op.shape[2]-1) // 2)), mode='edge')
    padded_aemc_stack = np.pad(aemc_stack, pad_width=(((element_op.shape[0]-1) // 2, (element_op.shape[0]-1) // 2), 
                                                    ((element_op.shape[1]-1) // 2, (element_op.shape[1]-1) // 2), 
                                                    ((element_op.shape[2]-1) // 2,(element_op.shape[2]-1) // 2)), mode='edge')
    core_seed = np.zeros(shape=padded_aemc_stack.shape, dtype=np.uint16)
    bin_roi =  binary_dilation(padded_aemc_stack > 0, structure=np.ones(shape=(3, 3, 3)))
    label_roi = measure.label(bin_roi, connectivity=1)
    label_roi_properties = measure.regionprops(label_roi)
    current_id = 0
    seed_id = 0
    draw_id = 0
    for prop in label_roi_properties:
        current_id = current_id + 1
        current_bbox = prop.bbox
        padded_aemc_roi_roi = (label_roi[current_bbox[0]:current_bbox[3]+1, 
                                        current_bbox[1]:current_bbox[4]+1,
                                        current_bbox[2]:current_bbox[5]+1] == current_id).astype(np.uint8)
        padded_aemc_roi_roi = padded_aemc_roi_roi * padded_aemc_stack[current_bbox[0]:current_bbox[3]+1, 
                                                                      current_bbox[1]:current_bbox[4]+1,
                                                                      current_bbox[2]:current_bbox[5]+1]
        if 4 in padded_aemc_roi_roi and len(np.unique(padded_aemc_roi_roi)) > 3:
            # single area is available
            padded_aemc_roi_c = (padded_aemc_roi_roi == 4).astype(np.uint8)
            padded_aemc_roi_c = binary_dilation(padded_aemc_roi_c, np.ones(shape=(3, 5, 5)))
            seed_label_roi = measure.label(padded_aemc_roi_c, connectivity=1)
            seed_label_properties = measure.regionprops(seed_label_roi)
            current_seed_id = 0
            seed_node_list = []
            seed_index_dict = {}
            seed_node_area_dict = {}
            seed_node_array_dict = {}
            seed_node_distance_dict = {}
            seed_node_z_dict = {}
            for seed_label_prop in seed_label_properties:
                current_seed_id += 1
                if seed_label_prop.area >= 4:
                    seed_id = seed_id + 1
                    seed_node_array = (seed_label_roi == current_seed_id).astype(np.uint8)
                    seed_node_array_dict[seed_id] = seed_node_array
                    seed_index = np.where(seed_label_roi == current_seed_id)
                    seed_node_list.append(seed_id)
                    seed_index_dict[seed_id] = seed_index
                    seed_node_area_dict[seed_id] = seed_label_prop.area
                    # core_seed[current_bbox[0]:current_bbox[3]+1,
                    #           current_bbox[1]:current_bbox[4]+1,
                    #           current_bbox[2]:current_bbox[5]+1][seed_index] = seed_id
            if len(seed_index_dict) < 2:
                for seed_id, seed_index in seed_index_dict.items():
                    if seed_node_area_dict[seed_id] > graph_thre:
                        draw_id = draw_id + 1
                        core_seed[current_bbox[0]:current_bbox[3]+1,
                                  current_bbox[1]:current_bbox[4]+1,
                                  current_bbox[2]:current_bbox[5]+1][seed_index] = draw_id
            else:
                for seed_id, seed_node_array in seed_node_array_dict.items():
                    distance_transform1 = distance_transform_edt(np.logical_not(seed_node_array), sampling=[ani_scale, 1, 1])
                    distance_transform2 = distance_transform_edt(np.logical_not(seed_node_array), sampling=[100, ani_scale, ani_scale])

                    seed_node_z_index = np.unique(np.where(seed_node_array)[0])
                    for z_layer_index in seed_node_z_index:
                        distance_transform1[z_layer_index] = distance_transform2[z_layer_index]
                    seed_node_distance_dict[seed_id] = distance_transform1
                    seed_node_z_dict[seed_id] = seed_node_z_index

                seed_edge_list = []
                for i, seed_single_node1 in enumerate(seed_node_list):
                    if i < len(seed_node_list):
                        
                        for seed_single_node2 in seed_node_list[i+1:]:
                            node1in2 = seed_node_array_dict[seed_single_node1] * seed_node_distance_dict[seed_single_node2]
                            node2in1 = seed_node_array_dict[seed_single_node2] * seed_node_distance_dict[seed_single_node1]
                            nose1in2_list = []
                            nose2in1_list = []
                            for seed_single_node1_z in seed_node_z_dict[seed_single_node1]:
                                # non_zero_elements
                                non_zero_elements = node1in2[seed_single_node1_z][node1in2[seed_single_node1_z] != 0]
                                nose1in2_list.append((1 - (1 / (1 + np.var(non_zero_elements)))) *  np.mean(non_zero_elements))
                            for seed_single_node2_z in seed_node_z_dict[seed_single_node2]:
                                # non_zero_elements
                                non_zero_elements = node2in1[seed_single_node2_z][node2in1[seed_single_node2_z] != 0]
                                nose2in1_list.append((1 - (1 / (1 + np.var(non_zero_elements)))) *  np.mean(non_zero_elements))
                            if np.min(np.array(nose1in2_list)) + np.min(np.array(nose2in1_list)) < coonnect_thre:
                                seed_edge_list.append((seed_single_node1, seed_single_node2))
                # create graph
                G = nx.Graph()
                G.add_nodes_from(seed_node_list)
                G.add_edges_from(seed_edge_list)
                subgraphs = list(nx.connected_components(G))
                for i, subgraph in enumerate(subgraphs):
                    connect_list = list(subgraph)
                    connect_index = (np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                    connect_area = 0
                    for connect_id in connect_list:
                        connect_area = connect_area + seed_node_area_dict[connect_id]
                        connect_index = (np.concatenate((connect_index[0], seed_index_dict[connect_id][0]), axis=0),
                                        np.concatenate((connect_index[1], seed_index_dict[connect_id][1]), axis=0),
                                        np.concatenate((connect_index[2], seed_index_dict[connect_id][2]), axis=0),)
                    if connect_area > graph_thre:
                        draw_id = draw_id + 1
                        core_seed[current_bbox[0]:current_bbox[3]+1,
                                  current_bbox[1]:current_bbox[4]+1,
                                  current_bbox[2]:current_bbox[5]+1][connect_index] = draw_id

    out_stack = watershed(-padded_aemc_stack, core_seed, mask=(padded_semantic_stack > 0))
    return out_stack[((element_op.shape[0]-1) // 2):-((element_op.shape[0]-1) // 2),
                     ((element_op.shape[1]-1) // 2):-((element_op.shape[1]-1) // 2),
                     ((element_op.shape[2]-1) // 2):-((element_op.shape[2]-1) // 2),]




def read_folder_down(source_path, down_scale=2, is_seg=False):
    if is_seg:
        inter_mode = cv2.INTER_NEAREST
    else:
        inter_mode = cv2.INTER_CUBIC
    file_list = os.listdir(source_path)
    file_list.sort(key=lambda x:int(x.split('.')[0]))
    save_stack = []
    for file_name in tqdm(file_list):
        img = io.imread(os.path.join(source_path, file_name))
        img =  cv2.resize(img, dsize=(img.shape[1]//down_scale, img.shape[0]//down_scale), interpolation=inter_mode)
        save_stack.append(img)
    save_stack = np.array(save_stack)
    return save_stack