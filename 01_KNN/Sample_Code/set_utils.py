from get_data_file import *
from numpy import dot
from numpy import shape



def rgb2gray(image):
    return dot(image[...,:3], [0.299, 0.587, 0.114])


def fetch_sets(source, actors, train=100, validation=10, test=10, invert=False):
    x_train = []
    x_validation = []
    x_test = []

    y_train = []
    y_validation = []
    y_test = []
    total = train+validation+test

    if invert:
        list_actors = fetch_actors(source)
        actors = [actor for actor in list_actors if actor not in actors]

    for actor in actors:
        faces = fetch_data(source, [actor], train+validation+test, threadtimeout=3)
        num_points = len(faces)
        faces
        if num_points >= total:
            for i in range(0,train):
                x_train.append(faces[i])
                y_train.append(actor)
            for i in range(train,train+validation):
                x_validation.append(faces[i])
                y_validation.append(actor)
            for i in range(train+validation,train+validation+test):
                x_test.append(faces[i])
                y_test.append(actor)
        else:
            raise ValueError('Not enough data to produce sets of %s - %d needed, %d found' %(actor, total, num_points))

    return x_train, y_train, x_validation, y_validation, x_test, y_test
