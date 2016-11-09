import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from vgg16 import vgg16
from scipy.misc import imsave


def render_naive(net_layer, net_input, img, sess):
    obj = tf.reduce_mean(net_layer)
    grad = tf.gradients(obj, net_input)[0]

    for _ in xrange(300):
        g = sess.run(grad, {net_input: [img]})[0]
        g /= g.std() + 1e-8
        img += g * 1.
        img = np.clip(img, 0, 255)
    return (img - img.mean()) / max(img.std(), 1e-4) * 0.1 + 0.5


if __name__ == "__main__":
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    img = np.random.uniform(size=(224, 224, 3)) + 100.0
    img = render_naive(vgg.conv4_1[:,:,:,370], imgs, img, sess)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
