from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Model
from keras.layers import Dense, Lambda, Activation, Input
from keras.layers import BatchNormalization
from keras.layers import add, concatenate
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras import losses



def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cosine_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(input_dim, nn_dim=128):
    '''
    Base network for feature extraction.
    
    '''
    input = Input(shape=(input_dim, ))
    dense1 = Dense(nn_dim)(input)
    bn1 = BatchNormalization()(dense1)
    relu1 = Activation('relu')(bn1)

    dense2 = Dense(nn_dim)(relu1)
    bn2 = BatchNormalization()(dense2)
    res2 = add([relu1, bn2])
    relu2 = Activation('relu')(res2)    
    
    dense3 = Dense(nn_dim)(relu2)
    bn3 = BatchNormalization()(dense3)
    res3 = add([relu2, bn3])
    relu3 = Activation('relu')(res3) 

    feats = concatenate([relu3, relu2, relu1])
    bn5 = BatchNormalization()(feats)

    model = Model(inputs=input, outputs=bn5)

    return model


def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))

def create_network(input_dim, nn_dim=128):
    base_network = create_base_network(input_dim, nn_dim)
    
    input_left = Input(shape=(input_dim,))
    input_right = Input(shape=(input_dim,))

    # share weights between two branches of siamese network
    network_left = base_network(input_left)
    network_right = base_network(input_right)
    
    distance = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape)([network_left, network_right])
    
    model = Model(inputs=[input_left, input_right], outputs=distance)
    return model


def train(X_train, Y_train, X_test, Y_test, epochs=1, iterations=25):
    net = create_network(384)

    # TODO(parvoberoi): test with different optimizers and loss methods
    optimizer = Adam(lr=0.001)
    net.compile(loss=contrastive_loss, optimizer=optimizer, metrics=['accuracy'])

    for iteration in range(iterations):
        net.fit(
            [X_train[:,0,:], X_train[:,1,:]],
            Y_train,
            validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
            batch_size=128,
            epochs=epochs,
            shuffle=True,
        )

        # compute final accuracy on training and test sets
        pred = net.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
        test_accuracy = compute_accuracy(pred, Y_test)

        print('***** Accuracy on test set: %0.2f%%' % (100 * test_accuracy))