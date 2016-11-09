import numpy as np
import matplotlib.pyplot as plt

weights = np.load('vgg16_weights.npz')
layers = sorted(weights.keys())

conv1w = weights[layers[0]]
numfilters = np.shape(conv1w)[-1]

plt.suptitle('Conv1_1 Filters')
for i in range(numfilters):
    plt.subplot(8,8,i+1)
    plt.imshow(conv1w[:,:,:,i])
    plt.axis('off')
plt.savefig('weights.eps')
plt.show()
