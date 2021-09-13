from watermarkJPEG import *

import itertools
#import numpy as np
#import matplotlib.pyplot as plt
#import PIL.Image
#import tensorflow as tf



# From https://github.com/rshin/differentiable-jpeg with small modifications
def showarray(a, fmt='png'):
    import IPython.display
    from cStringIO import StringIO
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

# 1. RGB -> YCbCr
# https://en.wikipedia.org/wiki/YCbCr
def rgb_to_ycbcr(image):
    matrix = np.array([[65.481, 128.553, 24.966],
                       [-37.797, -74.203, 112.],
                       [112., -93.786, -18.214]], dtype=np.float32).T / 255
    shift = [16., 128., 128.]
    
    result = tf.tensordot(image, matrix, axes=1) + shift
    result.set_shape(image.shape.as_list())
    return result

def rgb_to_ycbcr_jpeg(image):
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]], dtype=np.float32).T
    shift = [0., 128., 128.]
    
    result = tf.tensordot(image, matrix, axes=1) + shift
    result.set_shape(image.shape.as_list())
    return result

# 2. Chroma subsampling
def downsampling_420(image):
    # input: batch x height x width x 3
    # output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    y, cb, cr = tf.split(image, 3, axis=3)
    cb = tf.nn.avg_pool(cb, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    cr = tf.nn.avg_pool(cr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.squeeze(y, axis=-1), tf.squeeze(cb, axis=-1), tf.squeeze(cr, axis=-1)

'''# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
# https://github.com/tensorflow/models/issues/6245
def image_to_patches(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    k = 8
    batch_size, height, width = image.shape.as_list()[0:3]
    
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]
    
    image_reshaped = tf.reshape(image, [batch_size, height // k, k, -1, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, -1, k, k])'''

# 4. DCT
def dct_8x8_ref(image):
    image = image - 128
    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2*x+1)*u*np.pi/16) * np.cos((2*y+1)*v*np.pi/16)
        result[u, v] = value
    alpha = np.array([1./np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    return result * scale

def dct_8x8(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2*x+1)*u*np.pi/16) * np.cos((2*y+1)*v*np.pi/16)
    alpha = np.array([1./np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = scale * tf.tensordot(image, tensor, axes=2)
    result.set_shape(image.shape.as_list())
    return result

'''# 5. Quantizaztion
y_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47],
                            [18, 21, 26, 66],
                            [24, 26, 56, 99],
                            [47, 66, 99, 99]]).T

def y_quantize(image, rounding, factor=1):
    image = image / (y_table * factor)
    image = rounding(image)
    return image
def c_quantize(image, rounding, factor=1):
    image = image / (c_table * factor)
    image = rounding(image)
    return image

# -5. Dequantization
def y_dequantize(image, factor=1):
    return image * (y_table * factor)
def c_dequantize(image, factor=1):
    return image * (c_table * factor)'''

# -4. Inverse DCT
def idct_8x8_ref(image):
    alpha = np.array([1./np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2*u+1)*x*np.pi/16) * np.cos((2*v+1)*y*np.pi/16)
        result[u, v] = value
    return result * 0.25 + 128

def idct_8x8(image):
    alpha = np.array([1./np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha
    
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2*u+1)*x*np.pi/16) * np.cos((2*v+1)*y*np.pi/16)
    result = 0.25 * tf.tensordot(image, tensor, axes=2) + 128
    result.set_shape(image.shape.as_list())
    return result

'''# -3. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = patches.shape.as_list()[0]
    image_reshaped = tf.reshape(patches, [batch_size, height // k, width // k, k, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, height, width])'''

# -2. Chroma upsampling
def upsampling_420(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x height x width x 3
    def repeat(x, k=2):
        height, width = x.shape.as_list()[1:3]
        x = tf.expand_dims(x, -1)
        x = tf.tile(x, [1, 1, k, k])
        x = tf.reshape(x, [-1, height * k, width * k])
        return x
    cb = repeat(cb)
    cr = repeat(cr)
    return tf.stack((y, cb, cr), axis=-1)

# -1. YCbCr -> RGB
def ycbcr_to_rgb(image):
    matrix = np.array([[298.082, 0, 408.583],
                       [298.082, -100.291, -208.120],
                       [298.082, 516.412, 0]], dtype=np.float32).T / 256
    shift = [-222.921, 135.576, -276.836]
    
    result = tf.tensordot(image, matrix, axes=1) + shift
    result.set_shape(image.shape.as_list())
    return result

def ycbcr_to_rgb_jpeg(image):
    matrix = np.array([[1., 0., 1.402],
                       [1, -0.344136, -0.714136],
                       [1, 1.772, 0]], dtype=np.float32).T
    shift = [0, -128, -128]
    
    result = tf.tensordot(image + shift, matrix, axes=1)
    result.set_shape(image.shape.as_list())
    return result

# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
# https://github.com/tensorflow/models/issues/6245
def image_to_patches(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    k = 8
    #batch_size, height, width = image.shape.as_list()[0:3]
    
    batch_size = tf.shape(image)[0]
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]
    
    image_reshaped = tf.reshape(image, [batch_size, height // k, k, width // k, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, (height * width) // (k*k) , k, k])

# -3. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    #batch_size = patches.shape.as_list()[0]
    batch_size = tf.shape(patches)[0]
    image_reshaped = tf.reshape(patches, [batch_size, height // k, width // k, k, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, height, width])

# 5. Quantizaztion
y_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47],
                            [18, 21, 26, 66],
                            [24, 26, 56, 99],
                            [47, 66, 99, 99]]).T

def y_quantize(image, dqt_y):
    image = image / dqt_y
    image = diff_round(image)
    return image
def c_quantize(image, dqt_c):
    image = image / dqt_c
    image = diff_round(image)
    return image

def y_quantize_tfround(image, dqt_y):
    image = image / dqt_y
    image = tf.round(image)
    return image
def c_quantize_tfround(image, dqt_c):
    image = image / dqt_c
    image = tf.round(image)
    return image

# -5. Dequantization
def y_dequantize(image, dqt_y):
    return image * dqt_y
def c_dequantize(image, dqt_c):
    return image * dqt_c



def jpeg_compress_decompress_layer(image, height, width, dqt_y, dqt_c,downsample_c=True):#input image should already be a tensor
    #image = tf.convert_to_tensor(image)
    #height, width = image.shape.as_list()[1:3]
    orig_height, orig_width = height, width
    if height % 16 != 0 or width % 16 != 0:
        # Round up to next multiple of 16
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width
        #top = vpad // 2
        #bottom = vpad - top
        #left = wpad // 2
        #right = wpad - left
        
        #image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
        image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')
    
    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    y, cb, cr = downsampling_420(image)
    #components = {'y': y, 'cb': cb, 'cr': cr}
    
    comp_y = y
    comp_y = image_to_patches(comp_y)
    comp_y = dct_8x8(comp_y)
    comp_y =  y_quantize(comp_y, dqt_y)
        
    comp_cb = cb
    comp_cb = image_to_patches(comp_cb)
    comp_cb = dct_8x8(comp_cb)
    comp_cb = c_quantize(comp_cb, dqt_c)
        
    comp_cr = cr
    comp_cr = image_to_patches(comp_cr)
    comp_cr = dct_8x8(comp_cr)
    comp_cr = c_quantize(comp_cr, dqt_c)
    
    comp_y = y_dequantize(comp_y, dqt_y)
    comp_y = idct_8x8(comp_y)
    comp_y = patches_to_image(comp_y, height, width)
    
    comp_cb = c_dequantize(comp_cb, dqt_c)
    comp_cb = idct_8x8(comp_cb)
    comp_cb = patches_to_image(comp_cb, int(height / 2), int(width / 2))
    
    comp_cr = c_dequantize(comp_cr, dqt_c)
    comp_cr = idct_8x8(comp_cr)
    comp_cr = patches_to_image(comp_cr, int(height / 2), int(width / 2))

    image = upsampling_420(comp_y, comp_cb, comp_cr)

    image = ycbcr_to_rgb_jpeg(image)
    
    image = image[:, 0:orig_height, 0:orig_width]

    image = tf.minimum(255., tf.maximum(0., image))
    
    return image

def jpeg_compress_decompress_exact(image, height, width, dqt_y, dqt_c,downsample_c=True):#input image should already be a tensor
    #image = tf.convert_to_tensor(image)
    #height, width = image.shape.as_list()[1:3]
    orig_height, orig_width = height, width
    if height % 16 != 0 or width % 16 != 0:
        # Round up to next multiple of 16
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width
        #top = vpad // 2
        #bottom = vpad - top
        #left = wpad // 2
        #right = wpad - left
        
        #image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
        image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')
    
    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    y, cb, cr = downsampling_420(image)
    #components = {'y': y, 'cb': cb, 'cr': cr}
    
    comp_y = y
    comp_y = image_to_patches(comp_y)
    comp_y = dct_8x8(comp_y)
    comp_y =  y_quantize_tfround(comp_y, dqt_y)
        
    comp_cb = cb
    comp_cb = image_to_patches(comp_cb)
    comp_cb = dct_8x8(comp_cb)
    comp_cb = c_quantize_tfround(comp_cb, dqt_c)
        
    comp_cr = cr
    comp_cr = image_to_patches(comp_cr)
    comp_cr = dct_8x8(comp_cr)
    comp_cr = c_quantize_tfround(comp_cr, dqt_c)
    
    comp_y = y_dequantize(comp_y, dqt_y)
    comp_y = idct_8x8(comp_y)
    comp_y = patches_to_image(comp_y, height, width)
    
    comp_cb = c_dequantize(comp_cb, dqt_c)
    comp_cb = idct_8x8(comp_cb)
    comp_cb = patches_to_image(comp_cb, int(height / 2), int(width / 2))
    
    comp_cr = c_dequantize(comp_cr, dqt_c)
    comp_cr = idct_8x8(comp_cr)
    comp_cr = patches_to_image(comp_cr, int(height / 2), int(width / 2))

    image = upsampling_420(comp_y, comp_cb, comp_cr)

    image = ycbcr_to_rgb_jpeg(image)
    
    image = image[:, 0:orig_height, 0:orig_width]

    image = tf.minimum(255., tf.maximum(0., image))
    
    return image

def diff_round(x):
    return tf.round(x) + (x - tf.round(x))**3

def build_specific_model(height, width, base_model, pixel_shape = 224, enable_jpeg = True, quant_lumin = None, quant_uv = None):
    #if enable_jpeg and quant_lumin == None:
     #   print('need to input dqt')
    image_in = tf.keras.Input(shape=(height, width,3))
    if enable_jpeg:
        x = jpeg_compress_decompress_layer(image_in,height, width,quant_lumin, quant_uv)
        x = tf.image.resize(x, (224, 224))
    else:
        x = tf.image.resize(image_in, (224, 224))
    x = preprocess_input(x)
    outputs = base_model(x)
    model_jpeg_resize = keras.Model(image_in, outputs)
    
    logits_model = tf.keras.Model(base_model.input,base_model.layers[-1].output)
    logit_output = logits_model(x)
    model_jpeg_resize_logits = keras.Model(image_in, logit_output)
    
    return model_jpeg_resize, model_jpeg_resize_logits

def attack_image_no_downscale(fn_image, attack_type, model,logits_model, conf = 0):
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
    #x = downscale(img,target_size = (224,224))

    x = np.expand_dims(img, axis=0)
    tensor_image = tf.dtypes.cast(x, tf.float32)
    #x_ori = x.copy()
    #assert x_ori.max()>200
    #x_prep = preprocess_input(x)
    
    if attack_type == 'fgsm':
        adv_sample_prep = fgm_attack_untargetted(model, logits_model, tensor_image,fgm_step)
        
    if attack_type == 'pgd':   
        adv_sample_prep= pgd_attack_untargetted(model, logits_model, tensor_image, pgd_thresh, pgd_step, pgd_nb_step)
    
    if attack_type == 'cw':
        cw_param = {'batch_size':1, 'clip_max':255, 'clip_min':-255, 'confidence': conf}
        #tensor_image = tf.dtypes.cast(tensor_image, tf.float32)
        adv_sample_prep = carlini_wagner_l2(model, tensor_image, **cw_param)
        

    
    #temp = down
    
    return adv_sample_prep# the returned image is RG


def attack_image_no_downscale_np(image, attack_type, model,logits_model, conf = 0):
    pgd_thresh = 0.3
    pgd_step = 0.007843 # these param selections
    pgd_nb_step = 16
    fgm_step = 0.125
    assert attack_type in ['cw','fgsm','pgd']

    #img = cv2.imread(fn_image).astype('float32')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    tensor_image = tf.convert_to_tensor(image)
    tensor_image = tf.dtypes.cast(tensor_image, tf.float32)
    
    if attack_type == 'fgsm':
        adv_sample_prep = fgm_attack_untargetted(model, logits_model, tensor_image,fgm_step)
        
    if attack_type == 'pgd':   
        adv_sample_prep= pgd_attack_untargetted(model, logits_model, tensor_image, pgd_thresh, pgd_step, pgd_nb_step)
    
    if attack_type == 'cw':
        cw_param = {'batch_size':1, 'clip_max':255, 'clip_min':-255, 'confidence': conf}
        #tensor_image = tf.dtypes.cast(tensor_image, tf.float32)
        adv_sample_prep = carlini_wagner_l2(model, tensor_image, **cw_param)
        

    
    #temp = down
    
    return adv_sample_prep# the returned image is RG







