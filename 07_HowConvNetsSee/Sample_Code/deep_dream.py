from sys import stdout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave
from vgg16 import vgg16


def tiled_grad(img, net_grad, net_input, sess, tile=224):
    h, w = img.shape[:2]
    sx, sy = np.random.randint(tile, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            sub = img_shift[y:y + tile, x:x + tile]
            g = sess.run(net_grad, {net_input: [sub]})
            grad[y:y + tile, x:x + tile] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def deepdream(net_layer, net_input, img, sess, n_it=30, step=0.5, n_oct=7, sc_oct=1.4):
    obj = tf.square(net_layer)
    act_map = net_layer / tf.reduce_max(net_layer, [0,1])
    grad = tf.gradients(obj, net_input, grad_ys=act_map)[0]

    octaves = []
    for i in range(n_oct - 1):
        hw = img.shape[:2]
        lo = imresize(img, np.int32(np.float32(hw) / sc_oct))
        hi = img - imresize(lo, hw)
        img = lo
        octaves.append(hi)

    print " OCT - ITER"
    for octave in range(n_oct):
        if octave > 0:
            hi = octaves[-octave]
            img = imresize(img, hi.shape[:2]) + hi
        img = img.astype('float32')
        for i in range(n_it):
            stdout.write("\r%3d%% - %3d%%" %
                         (100. * octave / n_oct, 100. * i / n_it))
            stdout.flush()
            g = tiled_grad(img, grad, net_input, sess)
            np.add(img, g * (step / (np.abs(g).mean() + 1e-14)), out=img)
            img = np.clip(img, 0, 255)

    img /= 255.
    img = np.uint8(np.clip(img, 0, 1) * 255.)
    stdout.write("\r100% - 100%\n")
    return img

if __name__ == "__main__":
    sess = tf.Session()
    with sess.as_default():
        imgs = tf.placeholder(
            tf.float32, [None, 224, 224, 3], name="img_input")
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
        img = imread('sky.jpg', mode='RGB').astype('float32')
        img = deepdream(vgg.conv5_2, imgs, img, sess)
        imsave('deepdream.jpg', img)
        plt.imshow(img)
        plt.show()
