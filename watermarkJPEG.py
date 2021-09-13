import numpy as np
# from PIL import Image # remove pillow dependency. Mixing PIL with cv2 is bad
from PIL import Image # only use for get quantization table

import matplotlib.pyplot as plt
import pywt
import os
from scipy.fftpack import dct
from scipy.fftpack import idct
from tensorflow import keras

from scipy.stats import entropy
import copy
import cv2
import time
from pywt import dwt2,idwt2

from math import log10, sqrt 

from tensorflow.keras.preprocessing import image

import tensorflow as tf
#import numpy as np
#from PIL import Image 
#import matplotlib.pyplot as plt
#import pywt
#import os
#from scipy.fftpack import dct
#from scipy.fftpack import idct
#from tensorflow import keras
import math

#from scipy.stats import entropy
#import copy
#import cv2
#import time
#from cleverhans.attacks import BasicIterativeMethod, FastGradientMethod, CarliniWagnerL2,DeepFool
#from cleverhans.utils_keras import KerasModelWrapper
import random
import cleverhans
from cleverhans.future.tf2.attacks import fast_gradient_method, basic_iterative_method, projected_gradient_descent, spsa, carlini_wagner_l2
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True) 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3

# from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

#import xml.dom.minidom
#import json



def n2label(love):
    class_idx = json.load(open("imagenet_class_index.json"))
    for k in range(len(class_idx)):
        if class_idx[str(k)][0] == love:
            return k
def tong(a,b):
    result = 0
    assert len(a)==len(b)
    for i in range(len(a)):
        if a[i]==b[i]:
            result += 1
    return result

def load_image_get_pred(file_path,pixel_shape = 224,nb_images = 100):# input like 'C:/Users/Chen/Desktop/watermarking/6_16_tf2_pipe/dataset/'
    #load ori images
    ori_image_set = np.zeros((nb_images,pixel_shape,pixel_shape,3))
    ori_image_set_prep = np.zeros((nb_images,pixel_shape,pixel_shape,3))
    for i in range(nb_images):
        #if (i%(nb_images//5) == 0):
         #   print(i/nb_images*100, '% was finished')
        #fn_ori_image = 'dataset/ILSVRC2012_val_0000000'+str(i+1)+'.JPEG'
        fn_ori_image = 'ILSVRC2012_val_0000'
        for j in range(4-len(str(i+1))):
            fn_ori_image += '0'
        fn_ori_image += str(i+1) +'.JPEG' 
        #print(fn_ori_image)

        img = image.load_img(file_path + fn_ori_image, target_size=(pixel_shape, pixel_shape))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x_ori = x.copy()
        #assert x_ori.max()>200
        x_prep = preprocess_input(x)

        ori_image_set[i] = x_ori
        ori_image_set_prep[i] = x_prep
    assert ori_image_set.min() >= 0 and ori_image_set_prep.min()<0
    
    return np.argmax(model.predict(ori_image_set_prep),axis = 1)

def upscale(img, target_size = (500,375)):
    ori_size = img.shape
    upscaled_img = np.zeros((target_size[0],target_size[1],3))
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            pixel_center_x = (i+0.5)/target_size[0]
            pixel_center_y = (j+0.5)/target_size[1]    # find the location of pixel center of upscaled img pixel
            upscaled_img[i][j] = img[math.floor(pixel_center_x*ori_size[0])][math.floor(pixel_center_y*ori_size[1])]
            # find the corresponding pixel in ori img
    return upscaled_img
    
def downscale(img, target_size = (224,224)):# only get the 
    ori_size = img.shape
    #if ori_size[0] < target_size[0] or ori_size[1] < target_size[1]:
        #print('small image may be in correct')
    downscaled_img = np.zeros((target_size[0],target_size[1],3))
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            pixel_left_x = i/target_size[0]
            pixel_up_y = j/target_size[1]
            # 找第一个大于等于boundry的pixel center
            downscaled_img[i][j] = img[math.ceil(pixel_left_x*ori_size[0]-0.5)][math.ceil(pixel_up_y*ori_size[1]-0.5)]
    return downscaled_img


def fgm_attack_targetted(model_used, logits_model, watermarked_image_prep, eps, clip_min = -1):
    # the name of it are pgd because I would like to make as little change to the original code as possible
    nb_images_tmp = nb_images
    pgd_attack_success = np.zeros(nb_images_tmp)
    #pgd_adv_sample = np.zeros((nb_images,pixel_shape,pixel_shape,3))#.astype('int')
    pgd_adv_sample_prep = np.zeros((nb_images_tmp,pixel_shape,pixel_shape,3))
    time_start = time.time()
    for i in range(nb_images_tmp):
        if (i%(nb_images_tmp//5) == 0):
            print(i/nb_images_tmp*100, '% was finished')
        ori_image_test = watermarked_image_prep[i:i+1]
        #ori_label, tmp = topn(ori_image_test,model_used, 1)
        target = np.random.randint(0,1000)
        #if target == ori_label and target > 100:
        #    target = target - 100
        #elif target == ori_label and target <= 100:
        #    target = target + 100
        target_label = np.reshape(target, (1,)).astype('int64')
        adv_example_targeted_label = fast_gradient_method(logits_model, ori_image_test, 
                                                          eps, np.inf, y=target_label, clip_min = clip_min, clip_max = 1, targeted=True)
        pgd_adv_sample_prep[i] = adv_example_targeted_label
        #pgd_adv_sample[i] = (adv_example_targeted_label+1)*127.5
        #adv_label, tmp = topn(adv_example_targeted_label,model_used, 1)
        #if ori_label[0]!=adv_label[0]:
         #   pgd_attack_success[i] = 1
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
    print(100*sum(pgd_attack_success)/nb_images_tmp, '% of attack successfully changed top-1 prediction')
    assert pgd_adv_sample_prep.max()<=1 and pgd_adv_sample_prep.min()>=-1
    return pgd_adv_sample_prep, pgd_attack_success

def pgd_attack_targetted(model_used, logits_model_used, watermarked_image_prep, eps, eps_iter, nb_iter, clip_min = -1):
    pgd_attack_success = np.zeros(nb_images)
    #pgd_adv_sample = np.zeros((nb_images,pixel_shape,pixel_shape,3))#.astype('int')
    pgd_adv_sample_prep = np.zeros((nb_images,pixel_shape,pixel_shape,3))
    time_start = time.time()
    for i in range(nb_images):
        if (i%(nb_images//5) == 0):
            print(i/nb_images*100, '% was finished')
        ori_image_test = watermarked_image_prep[i:i+1]
        ori_label, tmp = topn(ori_image_test,model_used, 1)
        target = np.random.randint(0,1000)
        if target == ori_label and target > 100:
            target = target - 100
        elif target == ori_label and target <= 100:
            target = target + 100
        target_label = np.reshape(target, (1,)).astype('int64')
        adv_example_targeted_label = projected_gradient_descent(logits_model_used, ori_image_test, eps, eps_iter, nb_iter,
                                                                np.inf,clip_min=clip_min, clip_max=1, 
                                                                y=target_label, targeted=True,sanity_checks=False)
        pgd_adv_sample_prep[i] = adv_example_targeted_label
        #pgd_adv_sample[i] = (adv_example_targeted_label+1)*127.5
        adv_label, tmp = topn(adv_example_targeted_label,model_used, 1)
        if ori_label[0]!=adv_label[0]:
            pgd_attack_success[i] = 1
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
    print(100*sum(pgd_attack_success)/nb_images, '% of attack successfully changed top-1 prediction')
    assert pgd_adv_sample_prep.max()<=1 and pgd_adv_sample_prep.min()>=-1
    return pgd_adv_sample_prep, pgd_attack_success


def fgm_attack_untargetted(model_used, logits_model, watermarked_image_prep, eps, clip_min = 0):
    nb_images_tmp = 1
    #pgd_attack_success = np.zeros(nb_images_tmp)
    #pgd_adv_sample = np.zeros((nb_images_tmp,pixel_shape,pixel_shape,3))#.astype('int')
    #pgd_adv_sample_prep = np.zeros((nb_images_tmp,pixel_shape,pixel_shape,3))
    for i in range(nb_images_tmp):
        #if (i%(nb_images_tmp//5) == 0):
         #   print(i/nb_images_tmp*100, '% was finished')
        ori_image_test = watermarked_image_prep[i:i+1]
        #ori_label, tmp = topn(ori_image_test,model_used, 1)
        adv_example_targeted_label = fast_gradient_method(logits_model, ori_image_test, 
                                                          eps, np.inf, clip_min = clip_min, clip_max = 255, targeted=False)
        #pgd_adv_sample_prep[i] = adv_example_targeted_label
        #pgd_adv_sample[i] = (adv_example_targeted_label+1)*127.5
        #adv_label, tmp = topn(adv_example_targeted_label,model_used, 1)
        #if ori_label[0]!=adv_label[0]:
         #   pgd_attack_success[i] = 1
    return adv_example_targeted_label


def pgd_attack_untargetted(model_used, logits_model_used, watermarked_image_prep, eps, eps_iter, nb_iter, clip_min = 0):
    for i in range(1):
        ori_image_test = watermarked_image_prep[i:i+1]
        adv_example_targeted_label = projected_gradient_descent(logits_model_used, ori_image_test, eps, eps_iter, nb_iter,
                                                                np.inf,clip_min=clip_min, clip_max=255, 
                                                                sanity_checks=False)
        #pgd_adv_sample[i] = (adv_example_targeted_label+1)*127.5
    return adv_example_targeted_label

def attack_image(fn_image, attack_type,model,logits_model,quant = 11, pixel_shape = 224, conf = 0):
    pgd_thresh = 0.3
    pgd_step = 0.007843
    pgd_nb_step = 16
    fgm_step = 0.125
    assert attack_type in ['cw','fgsm','pgd']
    #img = image.load_img(fn_image, target_size=(pixel_shape, pixel_shape))# 这里略微有点问题，可能要读原始数据，然后再用downscale，但是现在先不管，看看结果怎么样
    #x = image.img_to_array(img)
    img = cv2.imread(fn_image).astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = np.load('img_prefix+fn_ori_image+)
    downscaled_img = downscale(img,target_size = (pixel_shape,pixel_shape))
    
    x = np.expand_dims(downscaled_img, axis=0)
    x_ori = x.copy()
    #assert x_ori.max()>200
    x_prep = preprocess_input(x)
    
    if attack_type == 'fgsm':
        adv_sample_prep = fgm_attack_untargetted(model, logits_model, x_prep,fgm_step)
        
    if attack_type == 'pgd':   
        adv_sample_prep= pgd_attack_untargetted(model, logits_model, x_prep, pgd_thresh, pgd_step, pgd_nb_step)
    
    if attack_type == 'cw':
        cw_param = {'batch_size':1, 'clip_min':-1, 'confidence': conf}
        cw_tensor_image = tf.dtypes.cast(x_prep, tf.float32)
        adv_sample_prep = carlini_wagner_l2(model, cw_tensor_image, **cw_param)
        
    adversarial_perturbation = adv_sample_prep - x_prep      
    ori_image = cv2.imread(fn_image).astype('float32')
    ori_image_RGB = cv2.cvtColor(ori_image/255, cv2.COLOR_BGR2RGB)*255
    # img_YUV = cv2.cvtColor(ori_image/255, cv2.COLOR_BGR2YUV)*255
    # print('ori_image',extract_and_validate(img_YUV,quant))
    img_shape = ori_image.shape
    if attack_type != 'cw':
        adversarial_perturbation = adversarial_perturbation.numpy()
    adversarial_perturbation = adversarial_perturbation.squeeze()
    upscaled_adversarial_perturbation = upscale(adversarial_perturbation, target_size = (img_shape[0],img_shape[1]))    
    attacked_image = ori_image_RGB + upscaled_adversarial_perturbation*127.5
    
    #temp = down
    
    return attacked_image# the retirned image is RGB
        

def predict_fn(fn_image):
    assert fn_image[-4:] in ['.npy','jpeg','JPEG']
    if fn_image[-4:] == '.npy':
        img = np.load(fn_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        downscaled_img = downscale(img, target_size = (pixel_shape,pixel_shape))
        x = np.expand_dims(downscaled_img, axis=0)
        x_ori = x.copy()
        #assert x_ori.max()>200
        x_prep = preprocess_input(x)
        pred = np.argmax(model.predict(x_prep),axis = 1)
    else:
        img = cv2.imread(fn_image).astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        downscaled_img = downscale(img, target_size = (pixel_shape,pixel_shape))
        x = np.expand_dims(downscaled_img, axis=0)
        x_ori = x.copy()
        x_prep = preprocess_input(x)
        pred = np.argmax(model.predict(x_prep),axis = 1)
    return pred

def load_image_get_pred(file_path,pixel_shape = 224):# input like 'C:/Users/Chen/Desktop/watermarking/6_16_tf2_pipe/dataset/'
    #load ori images
    ori_image_set = np.zeros((nb_images,pixel_shape,pixel_shape,3))
    ori_image_set_prep = np.zeros((nb_images,pixel_shape,pixel_shape,3))
    for i in range(nb_images):
        #if (i%(nb_images//5) == 0):
         #   print(i/nb_images*100, '% was finished')
        #fn_ori_image = 'dataset/ILSVRC2012_val_0000000'+str(i+1)+'.JPEG'
        fn_ori_image = 'ILSVRC2012_val_0000'
        for j in range(4-len(str(i+1))):
            fn_ori_image += '0'
        fn_ori_image += str(i+1) +'.JPEG' 
        #print(fn_ori_image)

        img = image.load_img(file_path + fn_ori_image, target_size=(pixel_shape, pixel_shape))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x_ori = x.copy()
        #assert x_ori.max()>200
        x_prep = preprocess_input(x)

        ori_image_set[i] = x_ori
        ori_image_set_prep[i] = x_prep
    assert ori_image_set.min() >= 0 and ori_image_set_prep.min()<0
    
    return np.argmax(model.predict(ori_image_set_prep),axis = 1)

def y_quantization(img_Y, qf = 80):
    dqt_y = get_default_dqt_Y_by_qf(qf)
    quantized_img_Y = np.zeros((img_Y.shape[0],img_Y.shape[1]))

    for i in range(0,int(np.floor(img_Y.shape[0]/8)*8),8):
        for j in range(0,int(np.floor(img_Y.shape[1]/8)*8),8):
            current_block = img_Y[i:i+8,j:j+8]
            block_dct = cv2.dct(current_block)
            block_dct = np.round(block_dct/dqt_y)
            block_dct = block_dct*dqt_y

            new_block = cv2.idct(block_dct)
            quantized_img_Y[i:i+8,j:j+8] = new_block
    return quantized_img_Y

def L2_batch(ori,mod):
    assert ori.shape == mod.shape
    total_L2 = 0
    for i in range(ori.shape[0]):
        total_L2 += sqrt(np.sum((mod[i,:,:,:]-ori[i,:,:,:])**2))
    return total_L2/ori.shape[0]


def count_dct_change_rate(fn_ori,fn_atk):
    total_dct_change = 0
    total_blocks = 0
    total_value_change = 0
    
    quant_lumin, quant_uv = get_quantization_table(fn_atk)
    ori_image = cv2.imread(fn_ori).astype('float32')
    
    
    attacked_image = cv2.imread(fn_atk).astype('float32')
    #attacked_image = np.load(dump_location + 'wm_image_cw_attacked_conf'+str(conf)+'/'+all_cnn+'/'+ fn_ori_image + '_cw_attacked.npy')
    #attacked_image = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2RGB)
    #assert attacked_image.max()>1.5
    
    #assert ori_image.shape() == attacked_image.shape()
    ori_img_YUV = cv2.cvtColor(ori_image/255, cv2.COLOR_BGR2YUV)*255
    attacked_img_YUV = cv2.cvtColor(attacked_image/255, cv2.COLOR_BGR2YUV)*255
    #ori_img_Y = ori_img_YUV[:,:,0]
    #attacked_img_Y = attacked_img_YUV[:,:,0]
    
    
    #current_total_diff = np.zeros((8,8))
    for i in range(0,int(np.floor(ori_img_YUV.shape[0]/8)*8),8):
        for j in range(0,int(np.floor(ori_img_YUV.shape[1]/8)*8),8):
            for k in range(3):
                if k == 0:
                    dqt = quant_lumin
                else:
                    dqt = quant_uv
                    
                current_block1 = ori_img_YUV[i:i+8,j:j+8,k].copy()
                current_block2 = attacked_img_YUV[i:i+8,j:j+8,k].copy()

                block_dct1 = cv2.dct(current_block1)
                block_dct2 = cv2.dct(current_block2)
                total_blocks += 1
                #current_total_blocks += 1
                for dct_x in range(8):
                    for dct_y in range(8):
                        #total_diff[dct_x,dct_y] += abs(block_dct1[dct_x,dct_y] - block_dct2[dct_x,dct_y])
                        #current_total_diff[dct_x,dct_y] += abs(block_dct1[dct_x,dct_y] - block_dct2[dct_x,dct_y])
                        temp1 = (block_dct1[dct_x,dct_y] + dqt[dct_x,dct_y]/2)//dqt[dct_x,dct_y]
                        temp2 = (block_dct2[dct_x,dct_y] + dqt[dct_x,dct_y]/2)//dqt[dct_x,dct_y]# quantized value
                        if (temp1 != temp2):
                            total_dct_change += 1
                            total_value_change += abs(temp1-temp2) * dqt[dct_x,dct_y]
                            #print(block_dct1[dct_x,dct_y],block_dct2[dct_x,dct_y])
                            #dct_change[dct_x,dct_y] += 1
        #print('only for this image:')
        #for dct_x in range(8):
        #    for dct_y in range(8):
        #        current_total_diff[dct_x,dct_y] /= current_total_blocks
        #print(current_total_diff)
    return total_dct_change/total_blocks/64, total_value_change/total_dct_change
#print('total_blocks',total_blocks)
#for dct_x in range(8):
#    for dct_y in range(8):
#        total_diff[dct_x,dct_y] /= total_blocks
#        dct_change[dct_x,dct_y] /= total_blocks

# 1*64 -> 8*8 by zigzag. used for quantization
def flat_zigzag_8x8(input64):
    assert len(input64) == 64
    zigzag = np.array((        [[0,   1,   5,  6,   14,  15,  27,  28],
        [2,   4,   7,  13,  16,  26,  29,  42],
        [3,   8,  12,  17,  25,  30,  41,  43],
        [9,   11, 18,  24,  31,  40,  44,  53],
        [10,  19, 23,  32,  39,  45,  52,  54],
        [20,  22, 33,  38,  46,  51,  55,  60],
        [21,  34, 37,  47,  50,  56,  59,  61],
        [35,  36, 48,  49,  57,  58,  62,  63]]))
    output8x8 = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            output8x8[i,j] = input64[zigzag[i,j]]
    return output8x8

def get_zigzag():
    zigzag = np.array((        [[0,   1,   5,  6,   14,  15,  27,  28],
        [2,   4,   7,  13,  16,  26,  29,  42],
        [3,   8,  12,  17,  25,  30,  41,  43],
        [9,   11, 18,  24,  31,  40,  44,  53],
        [10,  19, 23,  32,  39,  45,  52,  54],
        [20,  22, 33,  38,  46,  51,  55,  60],
        [21,  34, 37,  47,  50,  56,  59,  61],
        [35,  36, 48,  49,  57,  58,  62,  63]]))
    return zigzag

def zigzag_flat(input8x8):
    assert input8x8.shape == (8,8)
    zigzag = np.array((        [[0,   1,   5,  6,   14,  15,  27,  28],
    [2,   4,   7,  13,  16,  26,  29,  42],
    [3,   8,  12,  17,  25,  30,  41,  43],
    [9,   11, 18,  24,  31,  40,  44,  53],
    [10,  19, 23,  32,  39,  45,  52,  54],
    [20,  22, 33,  38,  46,  51,  55,  60],
    [21,  34, 37,  47,  50,  56,  59,  61],
    [35,  36, 48,  49,  57,  58,  62,  63]]))
    output64 = np.zeros(64)
    for i in range(64):
        for j in range(8):
            for k in range(8):
                if zigzag[j,k] == i:
                    output64[i] = input8x8[j,k]
    return output64
        

# return dqt for lumin and other two
def get_quantization_table(filename):
    love = Image.open(filename)
    quantization = getattr(love, 'quantization', None)
    return flat_zigzag_8x8(quantization[0]),flat_zigzag_8x8(quantization[1])

def find_zigzag(a):
    assert a>=1 and a<=64
    zigzag = np.array((        [[0,   1,   5,  6,   14,  15,  27,  28],
    [2,   4,   7,  13,  16,  26,  29,  42],
    [3,   8,  12,  17,  25,  30,  41,  43],
    [9,   11, 18,  24,  31,  40,  44,  53],
    [10,  19, 23,  32,  39,  45,  52,  54],
    [20,  22, 33,  38,  46,  51,  55,  60],
    [21,  34, 37,  47,  50,  56,  59,  61],
    [35,  36, 48,  49,  57,  58,  62,  63]]))
    
    for i in range(8):
        for j in range(8):
            if zigzag[i][j] == a:
                return i,j