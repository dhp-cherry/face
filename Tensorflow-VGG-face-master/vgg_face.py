import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def vgg_face(param_path, input_maps):

    data = loadmat(param_path)

    # read meta info
    meta = data['meta']#人名
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0]#描述
    normalization = meta['normalization']#正常化
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
    input_maps = tf.image.resize_images(input_maps, [image_size[0], image_size[1]])

    # read layer info
    layers = data['layers']#图层面板
    current = input_maps
    network = {}
    for layer in layers[0]:
        name = layer[0]['name'][0][0]
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer[0]['stride'][0][0]
            kernel, bias = layer[0]['weights'][0][0]
            # kernel = np.transpose(kernel, (1, 0, 2, 3))
            bias = np.squeeze(bias).reshape(-1)#z.reshape(-1)
                                                # array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
            conv = tf.nn.conv2d(current, tf.constant(kernel),
                                strides=(1, stride[0], stride[0], 1), padding=padding)
            current = tf.nn.bias_add(conv, bias)#tf.add的一个特例，bias必须是一维的，支持广播形式
            print(name, 'stride:', stride, 'kernel size:', np.shape(kernel))
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
            print(name)
        elif layer_type == 'pool':
            stride = layer[0]['stride'][0][0]
            pool = layer[0]['pool'][0][0]
            current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                     strides=(1, stride[0], stride[0], 1), padding='SAME')
            print(name, 'stride:', stride)
        elif layer_type == 'softmax':
            current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
            print(name)

        network[name] = current

    return network, average_image, class_names

