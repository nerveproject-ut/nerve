import numpy as np
import struct
import torch
import os
import math

def quantize_power_of_2(arr, bits):
    # Find the maximum absolute value in the array 
    max_val = np.max(np.abs(arr))
    # Calculate the exponent for the power of 2 step size     
    exponent = int(np.ceil(np.log2(max_val) - 2.807))
    # print(np.log2(7)) # 2.807354922057604
    # print(2 ** 2.807) # 6.9982781192465735

    # Calculate the step size as a power of 2 
    step = 2 ** exponent
    
    # Quantize the array by rounding to the nearest quantized value     
    integer_numbers = (np.round(arr / step)).astype(np.int32)
    
    # Option 1: force it to be smaller in the corner case
    if integer_numbers.max() > 7: 
        # GPT:
        # Find all elements with the value 8
        indices = np.where(integer_numbers == 8)
        # Change the values of elements to 7
        integer_numbers[indices] = 7
        print("Error! Force 8 to 7!") # this case should never happen
    # Option 2: exponent = int(np.ceil(np.log2(max_val) - log2(7))) up there
    
    quantized_arr = (integer_numbers * step).astype(np.float32)
    
    return exponent+127, integer_numbers, quantized_arr # +127 for directly being used for SENeCA

def binary(num):
    assert(isinstance(num, float))
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def cut_bottom_to_int(f):
    return (int(float_to_hex(f),0) >>16)

# def cut_bottom_to_hex(f):
#     int_val = (int(float_to_hex(f),0) >>16)
#     return hex(struct.unpack('<I', struct.pack('<i', int_val))[0])

# def print_int_binary(n, num_bit):
#     # Check if the number is negative
#     if n < 0:
#         # Convert the negative number to its two's complement representation
#         n = (1 << num_bit) + n

#     # Convert the number to binary representation and remove the prefix '0b'
#     binary = bin(n)[2:]

#     # Print the binary representation
#     print(binary)

def bf16_arr_to_uint16_arr(bf16_arr):
    uint16_arr = np.zeros_like(bf16_arr)
    
    for index_tuple, value in np.ndenumerate(bf16_arr):
        # print(index_tuple, value)
        uint16_arr[index_tuple] = cut_bottom_to_int(value)
        
        # print(value, uint16_arr[index_tuple].astype(np.uint16))

    return uint16_arr.astype(np.uint16)

def int4_arr_to_uint16_arr_fc(int4_arr):
    num_row = int4_arr.shape[0]
    num_column = int4_arr.shape[1]
    if num_column % 32 != 0:
        zeros_to_add = (math.floor(num_column / 32) + 1) * 32 - num_column
        zeros_mtrx = np.zeros([num_row, zeros_to_add]).astype(np.int32)
        int4_arr = np.concatenate([int4_arr, zeros_mtrx], axis=1)
        
    num_column = int4_arr.shape[1]
    # assert(num_column % 4 == 0)
    assert(num_column % 32 == 0)
    
    int4_arr_change_order = np.zeros_like(int4_arr)
    
    for col_id in range(num_column):
        group_32_idx = int(col_id)//32 # group_32_idx==0 indicates that the col_id is between 0~32,  group_idx==1 indicates that the col_id is between 32~64
        group_8_idx = int(col_id-group_32_idx*32)//8 # there are 4 group_8 in a group_32. group_8_idx==0 indicates that the col_id is between 32*group_32+0~7,  group_8_idx==1 indicates that the col_id is between 32*group_32+8~15.
        
        new_col_id = (col_id - 32*group_32_idx - 8*group_8_idx) * 4 + 32*group_32_idx + group_8_idx
        # print(col_id, group_32_idx, group_8_idx, new_col_id)
        int4_arr_change_order[:, new_col_id] = int4_arr[:, col_id] # int32
    
    uint16_arr = np.zeros_like(int4_arr)[:, :int(num_column/4)]
    for row_id in range(num_row): # combine every 4 int4 weights
        for col_id in range(int(num_column/4)):
            # print(int4_arr_change_order[row_id, col_id*4 + 0], int4_arr_change_order[row_id, col_id*4 + 1], int4_arr_change_order[row_id, col_id*4 + 2], int4_arr_change_order[row_id, col_id*4 + 3])
            # print_int_binary((int4_arr_change_order[row_id, col_id*4 + 0] & 0xF) << 12, 4)
            # print_int_binary((int4_arr_change_order[row_id, col_id*4 + 1] & 0xF) << 8, 4)
            # print_int_binary((int4_arr_change_order[row_id, col_id*4 + 2] & 0xF) << 4, 4)
            # print_int_binary((int4_arr_change_order[row_id, col_id*4 + 3] & 0xF), 4)
            
            combined_weight = ((int4_arr_change_order[row_id, col_id*4 + 0] & 0xF) << 12 | (int4_arr_change_order[row_id, col_id*4 + 1] & 0xF) << 8 | (int4_arr_change_order[row_id, col_id*4 + 2] & 0xF) << 4  | (int4_arr_change_order[row_id, col_id*4 + 3] & 0xF)) # & 0xF is to get the last 4 bit
            # print(combined_weight)
            # print_int_binary(combined_weight, 16)
            
            uint16_arr[row_id, col_id] = combined_weight
    
    # print(uint16_arr)
    return uint16_arr

def int4_arr_to_uint16_arr_conv(int4_arr):
    # print(int4_arr.shape)

    num_intput_c = int4_arr.shape[0]
    kernel_size = int4_arr.shape[1]
    assert(kernel_size == int4_arr.shape[2])
    num_output_c = int4_arr.shape[3]
    if num_output_c < 32:
        zeros_to_add = 32 - num_output_c
        zeros_mtrx = np.zeros([num_intput_c, kernel_size, kernel_size, zeros_to_add]).astype(np.int32)
        int4_arr = np.concatenate([int4_arr, zeros_mtrx], axis=3)
        
    num_output_c = int4_arr.shape[3]
    # assert(num_output_c % 4 == 0)
    assert(num_output_c % 32 == 0)
    
    int4_arr_change_order = np.zeros_like(int4_arr)
    
    for col_id in range(num_output_c):
        group_32_idx = int(col_id)//32 # group_32_idx==0 indicates that the col_id is between 0~32,  group_idx==1 indicates that the col_id is between 32~64
        group_8_idx = int(col_id-group_32_idx*32)//8 # there are 4 group_8 in a group_32. group_8_idx==0 indicates that the col_id is between 32*group_32+0~7,  group_8_idx==1 indicates that the col_id is between 32*group_32+8~15.
        
        new_col_id = (col_id - 32*group_32_idx - 8*group_8_idx) * 4 + 32*group_32_idx + group_8_idx
        # print(col_id, group_32_idx, group_8_idx, new_col_id)
        int4_arr_change_order[:, :, :, new_col_id] = int4_arr[:, :, :, col_id] # int32
    
    uint16_arr = np.zeros_like(int4_arr)[:, :, :, :int(num_output_c/4)]
    
    for input_c_id in range(num_intput_c): # combine every 4 int4 weights
        for kernel_row_id in range(kernel_size): # combine every 4 int4 weights
            for kernel_col_id in range(kernel_size): # combine every 4 int4 weights
                for output_c_id in range(int(num_output_c/4)):
                    combined_weight = ((int4_arr_change_order[input_c_id, kernel_row_id, kernel_col_id, output_c_id*4 + 0] & 0xF) << 12 | (int4_arr_change_order[input_c_id, kernel_row_id, kernel_col_id, output_c_id*4 + 1] & 0xF) << 8 | (int4_arr_change_order[input_c_id, kernel_row_id, kernel_col_id, output_c_id*4 + 2] & 0xF) << 4  | (int4_arr_change_order[input_c_id, kernel_row_id, kernel_col_id, output_c_id*4 + 3] & 0xF)) # & 0xF is to get the last 4 bit
                    # print(combined_weight)
                    # print_int_binary(combined_weight, 16)
                    
                    uint16_arr[input_c_id, kernel_row_id, kernel_col_id, output_c_id] = combined_weight
    
    # print(num_output_c, uint16_arr.shape)
    return uint16_arr

# def save_exp_to_txt(file_name, exp):
#     with open(file_name+'.txt', 'w') as f:
#         f.write(str(exp))

def save_weights_to_txt(file_name, param_arr_to_save):
    num_dims = len(param_arr_to_save.shape)
    print(file_name+'.txt', "num_dims:", num_dims)
    
    if num_dims == 3: # the only situation is the threshold of conv layer, each channel has its own threshold
        param_arr_to_save = param_arr_to_save.squeeze()
        num_dims = len(param_arr_to_save.shape)
        assert(num_dims == 1)
    
    if num_dims == 1:
        num_row = param_arr_to_save.shape[0]
        with open(file_name+'.txt', 'w') as f:
            f.write('{')
            for row_idx in range(num_row):
                f.write(str(param_arr_to_save[row_idx]))
                if row_idx != num_row - 1:
                    f.write(', ')
            f.write('};')
            
    elif num_dims == 2:
        num_row = param_arr_to_save.shape[0]
        num_col = param_arr_to_save.shape[1]
        num_col_to_write = num_col
        if num_col < 8:
            num_col_to_write = 8 # dirty fix
        with open(file_name+'.txt', 'w') as f:
            f.write('{\n')
            for row_idx in range(num_row):
                f.write('{')
                for col_idx in range(num_col_to_write):
                    if col_idx > num_col - 1:
                        f.write('0')
                    else: 
                        f.write(str(param_arr_to_save[row_idx, col_idx]))
                    if col_idx != num_col_to_write - 1:
                        f.write(', ')
                f.write('}')
                if row_idx != num_row - 1:
                    f.write(',')
                f.write('\n')
            f.write('};')
            
    elif num_dims == 4:
        # print("Conv weight shape:", param_arr_to_save.shape)
        num_input_channel = param_arr_to_save.shape[0]
        kernel_size_X = param_arr_to_save.shape[1]
        kernel_size_Y = param_arr_to_save.shape[2]
        num_output_channel = param_arr_to_save.shape[3]
        assert (kernel_size_X == kernel_size_Y)
        kernel_size = kernel_size_X # (kernel_size - 1) * 2 - idx
        
        with open(file_name+'.txt', 'w') as f:
            f.write('{\n')
            for in_c_idx in range(num_input_channel):
                f.write('{\n')
                for row_idx in range(kernel_size):
                    f.write('{\n')
                    for col_idx in range(kernel_size):
                        f.write('{')
                        for out_c_idx in range(num_output_channel):
                            
                            f.write(str(param_arr_to_save[in_c_idx, (kernel_size - 1) * 2 - (row_idx+1) - 1, (kernel_size - 1) * 2 - (col_idx+1) - 1, out_c_idx])) # for Kevin's conv layer. the top-left neuron is updated by the top-left weight. So need to use the actually bottom-right weight.
                            # f.write(str(param_arr_to_save[in_c_idx, row_idx, col_idx, out_c_idx]))
                            
                            if out_c_idx != num_output_channel - 1:
                                f.write(', ')
                        
                        f.write('}')
                        if col_idx != kernel_size - 1:
                            f.write(',')
                        f.write('\n')
                        
                    f.write('}')
                    if row_idx != kernel_size - 1:
                        f.write(',')
                    f.write('\n')
                
                f.write('}')
                if in_c_idx != num_input_channel - 1:
                    f.write(',')
                f.write('\n')
                
            f.write('};')

def load_quantized_weights_from_np(param_folder, network, device):
    for name, param in network.named_parameters():
        if 'image_preprocess' in name:
            continue 
        
        file_name = os.path.join(param_folder, name)
        file_complete_path = file_name + '.npy'
        if 'feature_net.' in name and '.weight' in name:
            if '0' in name: # for the first and last layer we do not need to quantize the weights to int4
                param_arr_bf16 = np.load(file_complete_path)
                param_arr = param_arr_bf16
            else:
                param_arr_int4 = np.load(file_complete_path)
                assert(param_arr_int4.max() <= 7 and param_arr_int4.min() >= -8)
                layer_idx = int(name.split('.')[1])
                layer_exp = int(np.load(param_folder + '/layer_exp.npy', allow_pickle=True)[layer_idx]) - 127
                param_arr = (param_arr_int4 * (2 ** layer_exp)).astype(np.float32)
            
        else:
            param_arr_bf16 = np.load(file_complete_path)
            param_arr = param_arr_bf16
            
        param.data =  torch.from_numpy(param_arr).to(device)
        
# def same_param_to_txt(network, param_arr_int4):
#     for name, param in network.named_parameters():
#         if '.weight' in name: 
#             param_arr_to_save = param_arr_int4 
#             # print(param_arr_to_save.shape) # (output_channels, input_channels, kernel_size, kernel_size)
#             if len(param_arr_to_save.shape) == 4:
#                 param_arr_to_save = np.transpose(param_arr_to_save, (1, 2, 3, 0)) # (input_channels, kernel_size, kernel_size, output_channels) # align with Kevin's SENeCA C weights.h
#                 param_arr_uint16 = int4_arr_to_uint16_arr_conv(param_arr_to_save)
                
#             else:
#                 assert(len(param_arr_to_save.shape) == 2)
#                 param_arr_to_save = np.transpose(param_arr_to_save, (1, 0)) # (num_input_neurons, num_output_neurons) align with Kevin's SENeCA C weights.h
#                 param_arr_uint16 = int4_arr_to_uint16_arr_fc(param_arr_to_save)
#             param_arr_to_save = param_arr_uint16
            
#             save_exp_to_txt(file_name+'_exp', layer_exp+127)
#         else:
#             param_arr_to_save = bf16_arr_to_uint16_arr(param_arr)
        
#         save_weights_to_txt(file_name, param_arr_to_save)
        

def evaluate_layer_sparsity(layer_activities):
    """evaluate the sparsity of each layer

    Args:
        layer_activities (list): list of layer activities
    """
    batch_layer_sparsity = []
    for act in layer_activities:
        dense_activation = torch.prod(torch.tensor(act.shape[1:]))
        non_zero_counts = torch.count_nonzero(act, dim=tuple(range(1, act.ndim)))
        act_sparsity = non_zero_counts / dense_activation
        batch_layer_sparsity.append([act_sparsity, non_zero_counts, dense_activation])

    return batch_layer_sparsity
