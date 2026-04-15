import torch
import torch.nn as nn
from nerve.training.yolo_train_utils.config import cfg
from nerve.training.yolo_train_utils.bbox_decode import decode
from nerve.training.yolo_train_utils.quat_utils import *


class QuantBF16(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, tensor):
        quantized_tensor = tensor.to(torch.bfloat16).to(torch.float32)
        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class Binarize(torch.autograd.Function): # same as a spike function
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        out = x.gt(0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()

        sg = 1 / (1 + width * x * x)
            
        # print(sg.size(), grad_input.size())
        return grad_input * sg, None


class FATReLU(nn.Module):
    """ FatReLU activation function """

    def __init__(self, threshold, width=torch.tensor(5.0)):
        """

        Args:
            threshold (tensor): threshold parameters
            width (tensor, optional): width of surrogate function. Defaults to torch.tensor(10.0).
        """
        super().__init__()
        self.threshold = threshold
        self.width = width
        self.binarize = Binarize.apply
    
    def forward(self, data):
        """forward function

        Args:
            data (tensor): input data

        Returns:
            tensor: output data
            tensor: data before performing fatrelu
        """
        data = torch.relu(data)
        data_pre_fatrelu = data.clone()
        activation_map = self.binarize(data - torch.abs(self.threshold), self.width)
        data = data * activation_map
        return data, data_pre_fatrelu

class VeryTinyYoloOneScale(nn.Module):
    """ Tiny-YOLO network with only one scales """

    def __init__(self, anchors, strides, device, threshold_img=0.0, fatrelu=True, init_threshold=1e-4):
        """
        :param anchors: information for anchors
        :param strides: list of strides
        """
        super(VeryTinyYoloOneScale, self).__init__()
        self.anchors = anchors
        self.strides = strides
        self.num_classes = 1 # we only detect human, scfg['YOLO']['CLASSES_NUM']
        self.anchor_per_scale = cfg['YOLO']['ANCHOR_PER_SCALE']
        self.threshold_img = threshold_img
        self.fatrelu = fatrelu
        self.init_threshold = init_threshold
        self.device = device
        self.quant_bf16 = QuantBF16.apply
        self.img_idx = 0

        self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.BN1 = nn.BatchNorm2d(16)
        self.ReLU1 = FATReLU(nn.Parameter((torch.ones(16, 1, 1)  + torch.rand(16, 1, 1))  * init_threshold)) if self.fatrelu else nn.ReLU()
        self.MP1 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.BN2 = nn.BatchNorm2d(32)
        self.ReLU2 = FATReLU(nn.Parameter((torch.ones(32, 1, 1)  + torch.rand(32, 1, 1))  * init_threshold)) if self.fatrelu else nn.ReLU()
        self.MP2 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.BN3 = nn.BatchNorm2d(64)
        self.ReLU3 = FATReLU(nn.Parameter((torch.ones(64, 1, 1)  + torch.rand(64, 1, 1))  * init_threshold)) if self.fatrelu else nn.ReLU()
        self.MP3 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.BN4 = nn.BatchNorm2d(128)
        self.ReLU4 = FATReLU(nn.Parameter((torch.ones(128, 1, 1)  + torch.rand(128, 1, 1))  * init_threshold)) if self.fatrelu else nn.ReLU()
        self.MP4 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.BN5 = nn.BatchNorm2d(128)
        self.ReLU5 = FATReLU(nn.Parameter((torch.ones(128, 1, 1)  + torch.rand(128, 1, 1))  * init_threshold)) if self.fatrelu else nn.ReLU()
        self.MP5 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        # self.BN6 = nn.BatchNorm2d(256)
        self.ReLU6 = FATReLU(nn.Parameter((torch.ones(256, 1, 1)  + torch.rand(256, 1, 1))  * init_threshold)) if self.fatrelu else nn.ReLU()
        self.conv7 = nn.Conv2d(256, self.anchor_per_scale * (self.num_classes + 5), kernel_size=(1, 1), stride=(1, 1))


    def param_quat(self, save_folder=None):
        
        def save_exp_to_txt(file_name, exp):
            with open(file_name+'.txt', 'w') as f:
                f.write(str(exp))
                
        self.save_folder = save_folder
            
        output_dict_quantized_params = {} # not used for now
        output_dict_quantized_params['layer_exp'] = [None] * 7 # for int4 quantization
        
        for name, param in self.named_parameters(): 
            # print(name, param.data.size())
            if "conv" in name and "weight" in name:
                param_array = param.data.cpu().detach().numpy()
                exponent, integer_numbers, quatized_arr = quantize_power_of_2(param_array, bits=4)
                param.data = torch.from_numpy(quatized_arr).to(self.device)
                assert(len(integer_numbers.shape) == 4)
                assert(integer_numbers.max() <= 7 and integer_numbers.min() >= -8)
                assert(exponent <= 127 and exponent >= -128)
                if save_folder is not None:
                    if integer_numbers.shape[2] != 1: # 3 x 3 conv
                        integer_numbers = np.transpose(integer_numbers, (1, 2, 3, 0)) # (input_channels, kernel_size, kernel_size, output_channels) # align with Kevin's SENeCA C weights.h
                        param_arr_uint16 = int4_arr_to_uint16_arr_conv(integer_numbers)
                    else: # 1 x 1 conv
                        integer_numbers = np.transpose(integer_numbers.squeeze(), (1, 0))
                        param_arr_uint16 = int4_arr_to_uint16_arr_fc(integer_numbers)
                    save_weights_to_txt(save_folder+'/'+name, param_arr_uint16)
                    save_exp_to_txt(save_folder+'/'+name+".exp", exponent)
            else:
                param.data = param.data.to(torch.bfloat16).to(torch.float32)
                if save_folder is not None:
                    param_array = param.data.cpu().detach().numpy()
                    param_arr_uint16 = bf16_arr_to_uint16_arr(quatized_arr)
                    save_weights_to_txt(save_folder+'/'+name, param_arr_uint16)
                    
        
    def log_image(self, image): # the first 16 bits: event channel index, rest 16 bits: pixel intensity in BF16.
        # if self.img_idx % self.log_freq != 0:
        #     return
        
        H = image.size()[2]
        W = image.size()[3]
        event_list = []
        
        for row_id in range(H):
            for column_id in range(W):
                intensity = image[0, 0, row_id, column_id]
                if intensity != 0:
                    # print(row_id, column_id, intensity)
                    intensity_BF16 = intensity.cpu().detach().numpy().astype(np.float32).item()
                    intensity_BF16_hex_str = float_to_hex(intensity_BF16)[2:6]
                    # print(intensity_BF16, intensity_BF16_hex_str)
                    
                    evheader_str = hex(1 << 20 | row_id << 10 | column_id)[2:] # align with Kevin and Sander # 1 is the number of event in this pixel location. the img is grey-scale so it is always one.  
                    # print(row_id, column_id)
                    
                    while True:
                        if len(evheader_str) < 8:
                            evheader_str = '0' + evheader_str
                        else:
                            break
                    evheader_hex_str = '0x'+evheader_str
                    assert(len(evheader_hex_str)==10)

                    event_list.append(evheader_hex_str)
                    event_list.append('0x0000'+intensity_BF16_hex_str)
   
        print("Total number of pixel to process in image No. {}: {}".format(self.img_idx, len(event_list)))
        
        # write event_list to file        
        with open(os.path.join(self.save_folder, 'img_ev.txt'), 'a') as fp: # 'a' 'w+'
            for item in event_list:
                # write each item on a new line
                fp.write("%s\n" % item)
            fp.write("0xa0a0a0a0\n")
            
            
    def print_events(self, output):
        # for validating the output of seneca
        # if self.img_idx % self.log_freq != 0:
        #     return
        
        non_zero_indexes = torch.nonzero(output, as_tuple=True)
        
        event_list = []
        for i in range (non_zero_indexes[0].size()[0]):
            event = np.array([[non_zero_indexes[0][i].data.cpu().numpy(), non_zero_indexes[1][i].data.cpu().numpy(), non_zero_indexes[2][i].data.cpu().numpy(), non_zero_indexes[3][i].data.cpu().numpy(), output[non_zero_indexes[0][i].data.cpu().numpy()][non_zero_indexes[1][i].data.cpu().numpy()][non_zero_indexes[2][i].data.cpu().numpy()][non_zero_indexes[3][i].data.cpu().numpy()].data.cpu().numpy()]])
            event_list.append(event)
        event_array = np.concatenate(event_list)
        # print(event_array)
        
        arrSortedIndex = np.lexsort((event_array[:, 1], event_array[:, 3], event_array[:, 2]))
        event_array_sorted = event_array[arrSortedIndex , :]
        num_ev = event_array_sorted.shape[0]
        
        # Set print options
        num_to_print = num_ev
        # if num_ev > 100:
        #     num_to_print = 100
        for ev_idx in range(num_to_print):
            print(", ".join(str(i) for i in event_array_sorted[ev_idx, 1:4].T.astype(int)), end=', ')
            print(event_array_sorted[ev_idx, 4])

        # for validating the output of seneca END
        
        dense_activation = torch.prod(torch.tensor(output.shape[1:]))
        non_zero_counts = torch.count_nonzero(output, dim=tuple(range(1, output.ndim)))
        act_sparsity = non_zero_counts / dense_activation
        print("Density =", act_sparsity)
        # print("size", output.size())

        return event_array_sorted
    

    def forward(self, input_data):
        """
        Forward function

        :param input_data: input data
        :return: [conv_high_res, conv_low_res], [pred_high_res, pred_low_res]
        """
        device = input_data.device
        
        # input_data = torch.relu(input_data - self.threshold_img)
        input_data = self.quant_bf16(input_data)
        
        # # visualize the thresholded image
        # image_np = input_data[0, :, :, :].cpu().squeeze().numpy()
        # image_np = (image_np * 255).astype(np.uint8)
        # image_to_save = Image.fromarray(image_np, "L")
        # image_filename = "thre_image.png"
        # image_to_save.save(image_filename)
        
        # # calculate the density
        # dense_activation = torch.prod(torch.tensor(input_data.shape[1:]))
        # non_zero_counts = torch.count_nonzero(input_data, dim=tuple(range(1, input_data.ndim)))
        # act_sparsity = non_zero_counts / dense_activation
        # print("Input Image Density =", act_sparsity)
        
        # log the image if needed
        # self.log_image(input_data)
        
        act_list = []
        pre_act_list = []
        
        act_list.append(input_data)

        x = self.conv1(input_data)
        x = self.quant_bf16(x)
        # x = self.BN1(x)
        # x = self.quant_bf16(x)
        if self.fatrelu:
            x, pre_act1 = self.ReLU1(x)
            pre_act_list.append(pre_act1)
        else:
            x = self.ReLU1(x)
        x = self.MP1(x)
        act_list.append(x)
        
        # self.print_events(x)
        
        x = self.conv2(x)
        x = self.quant_bf16(x)
        # x = self.BN2(x)
        # x = self.quant_bf16(x)
        if self.fatrelu:
            x, pre_act2 = self.ReLU2(x)
            pre_act_list.append(pre_act2)
        else:
            x = self.ReLU2(x)
        x = self.MP2(x)
        act_list.append(x)
        
        # self.print_events(x)
        
        x = self.conv3(x)
        x = self.quant_bf16(x)
        # x = self.BN3(x)
        # x = self.quant_bf16(x)
        if self.fatrelu:
            x, pre_act3 = self.ReLU3(x)
            pre_act_list.append(pre_act3)
        else:
            x = self.ReLU3(x)
        x = self.MP3(x)
        act_list.append(x)
        
        # self.print_events(x)
        
        x = self.conv4(x)
        x = self.quant_bf16(x)
        # x = self.BN4(x)
        # x = self.quant_bf16(x)
        if self.fatrelu:
            x, pre_act4 = self.ReLU4(x)
            pre_act_list.append(pre_act4)
        else:
            x = self.ReLU4(x)
        x = self.MP4(x)
        act_list.append(x)
        
        # self.print_events(x)
        
        x = self.conv5(x)
        x = self.quant_bf16(x)
        # x = self.BN5(x)
        # x = self.quant_bf16(x)
        if self.fatrelu:
            x, pre_act5 = self.ReLU5(x)
            pre_act_list.append(pre_act5)
        else:
            x = self.ReLU5(x)
        x = self.MP5(x)
        act_list.append(x)
        
        # self.print_events(x)
        
        x = self.conv6(x)
        x = self.quant_bf16(x)
        # x = self.BN6(x)
        # x = self.quant_bf16(x)
        if self.fatrelu:
            x, pre_act6 = self.ReLU6(x)
            pre_act_list.append(pre_act6)
        else:
            x = self.ReLU6(x)
        act_list.append(x)
        
        # self.print_events(x)
        
        conv_low_res = self.conv7(x)
        conv_low_res = self.quant_bf16(conv_low_res)
        
        # self.print_events(conv_low_res)

        conv_low_res, pred_low_res = decode(conv_low_res, device, self.anchors, self.strides, i=0)
        
        self.img_idx = self.img_idx + 1

        return [conv_low_res], [pred_low_res], pre_act_list, act_list # (pre_act1, pre_act2, pre_act3, pre_act4, pre_act5, pre_act6)

