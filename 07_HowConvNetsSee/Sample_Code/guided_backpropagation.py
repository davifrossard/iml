import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from vgg16 import vgg16, class_names
from scipy.misc import imread, imresize
from glob import glob
from gradient import get_saliency, load_images, plot_maps


@tf.RegisterGradient("GuidedRelu")
def _custom_relu(op, grad):
    relub = tf.cast(tf.greater(op.outputs[0], 0), tf.float32) # Only take activations > 0
    gradb = tf.cast(tf.greater(grad, 0), tf.float32) # Only propagate gradients > 0
    return relub * gradb * grad

if __name__ == "__main__":
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    imgs = load_images('images/')

    max_class, guided_grad = get_saliency(imgs, vgg, sess)
    for img, cl, gg in zip(imgs, max_class, guided_grad):
        plot_maps(img, cl, gg)
