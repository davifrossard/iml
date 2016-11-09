import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from vgg16 import vgg16, class_names
from scipy.misc import imread, imresize
from glob import glob

def plot_maps(img, cl, gg):
    plt.suptitle(class_names[cl], size=15, linespacing=2)
    plt.subplot(221)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")
    plt.subplot(222)
    plt.imshow((np.maximum(0, gg) / np.max(gg)))
    plt.axis('off')
    plt.title("Positive Saliency")
    plt.subplot(223)
    plt.imshow((np.maximum(0, -gg) / -np.min(gg)))
    plt.axis('off')
    plt.title("Negative Saliency")
    plt.subplot(224)
    plt.imshow(np.abs(gg).max(axis=-1), cmap='gray')
    plt.axis('off')
    plt.title("Absolute Saliency")
    plt.show()

def load_images(path):
    imgs = []
    for f in glob(path+'/*'):
        im = imread(f, mode='RGB')
        dims = np.array(np.shape(im)[:-1])
        r = 224./np.min(dims)
        dims = np.ceil(r*dims)
        im = imresize(im, dims.astype('int32'));
        left, top = np.ceil((dims-224.)/2.).astype('int32')
        right, bottom = np.ceil((dims+224.)/2.).astype('int32')
        im = im[left:right, top:bottom]
        imgs.append(im)
    return imgs

def get_saliency(imgs, network, sess=tf.Session()):
    with sess.as_default():
        classes = network.probs.eval({network.imgs: imgs})
        max_class = np.argmax(classes, 1)
        outputs_list = tf.unpack(network.fc3l, axis=1)
        grads = []
        for img, cl in zip(imgs, max_class):
            grad_tensor = tf.gradients(outputs_list[cl], network.imgs)
            grad = sess.run(grad_tensor, feed_dict={network.imgs: [img]})[0]
            grads.append(grad[0])
        return max_class, grads

if __name__ == "__main__":
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    imgs = load_images('images/')

    max_class, guided_grad = get_saliency(imgs, vgg, sess)
    for img, cl, gg in zip(imgs, max_class, guided_grad):
        plot_maps(img, cl, gg)
