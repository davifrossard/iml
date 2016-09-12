from set_utils import fetch_sets, rgb2gray
from k_nearest_neighbors import knn_classify
from get_data_file import fetch_actors
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

actors = list(fetch_actors("subset_actors.txt"))
actresses = list(fetch_actors("subset_actresses.txt"))
print "List of Actors:"
print '\n'.join(actors)
print "\nList of Actressess:"
print '\n'.join(actresses)

# Fetch dataset pictures
x_train_f, _, _, _, x_test_f, _ = fetch_sets("subset_actresses.txt", actresses, 40, 0, 20)
x_train_m, _, _, _, x_test_m, _ = fetch_sets("subset_actors.txt", actors, 40, 0, 20)

genders = ['Male', 'Female']
x_train = x_train_f + x_train_m
t_train = np.hstack((np.ones(len(x_train_f)), np.zeros(len(x_train_m))))

x_test = x_test_f + x_test_m
t_test = np.hstack((np.ones(len(x_test_f)), np.zeros(len(x_test_m))))

# Resize to 32x32 and convert to grayscale
x_train_bw = [rgb2gray(imresize(x, (32,32))) for x in x_train]
x_test_bw = [rgb2gray(imresize(x, (32,32))) for x in x_test]

# Classify pictures in test set
for xi in np.random.permutation(len(x_test_bw)):
    ti, nn = knn_classify(x_train_bw, t_train, x_test_bw[xi], 5)
    # Plot image with classification
    plt.subplot(121)
    plt.imshow(x_test[xi])
    plt.axis('off')
    plt.title(genders[int(ti)], color=('green' if ti == t_test[xi] else 'red'), weight='bold')
    
    # Plot nearest neighbors
    idx = [3,4,7,8,11]
    for i,nni in enumerate(nn):
        plt.subplot(3,4,idx[i])
        plt.imshow(x_train[nni])
        plt.axis('off')
        plt.title(genders[int(t_train[nni])])
        
    plt.show()
