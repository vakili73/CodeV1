# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:40:36 2018

@author: vvaki
"""


import keras.backend as K

def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    shape = int(y_pred.shape.as_list()[-1]/3)

    negative = y_pred[:, 0:shape]
    positive = y_pred[:, shape:2*shape]
    anchor = y_pred[:, 2*shape:3*shape]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=-1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=-1)

    # compute loss
    basic_loss = K.maximum(pos_dist-neg_dist+alpha ,0)
 
    return K.mean(basic_loss)


def contrastive_loss(y_true, y_pred, margin = 1):
    """
    Train a Siamese MLP on pairs of digits from the MNIST dataset.
    It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
    output of the shared network and by optimizing the contrastive loss (see paper
    for mode details).
    [1] "Dimensionality Reduction by Learning an Invariant Mapping"
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Gets to 99.5% test accuracy after 20 epochs.
    3 seconds per epoch on a Titan X GPU
    """

    shape = int(y_pred.shape.as_list()[-1]/2)

    neg_or_pos = y_pred[:, 0:shape]
    anchor = y_pred[:, shape:2*shape]

    euclidean_distance = K.sqrt(K.sum(K.square(neg_or_pos - anchor), axis=-1))

    basic_loss =  y_true * K.square(euclidean_distance) + (1 - y_true) * K.square(K.maximum(0., margin - euclidean_distance))

    return K.mean(basic_loss)


# def polarization_loss_v1(y_true, y_pred):
#     """
#     Implementation of the triplet loss function
#     Arguments:
#     y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor data
#             positive -- the encodings for the positive data (similar to anchor)
#             negative -- the encodings for the negative data (different from anchor)
#     Returns:
#     loss -- real number, value of the loss
#     """

#     shape = int(y_pred.shape.as_list()[-1]/2)

#     negative = y_pred[:, 0:shape]
#     positive = y_pred[:, shape:2*shape]

#     anchor_neg = y_true[:, 0, :]
#     anchor_pos = y_true[:, 1, :]

#     # distance between the anchor and the negative
#     neg_dist = K.sum(K.square(anchor_neg-negative),axis=-1)

#     # distance between the anchor and the positive
#     pos_dist = K.sum(K.square(anchor_pos-positive),axis=-1)

#     # compute loss
#     contrastive_projection_loss = K.square(K.batch_dot(positive, anchor_neg)) + K.square(K.batch_dot(negative, anchor_pos))
    
#     positive_agrement_projection_loss = K.square(K.abs(K.batch_dot(positive, anchor_pos)) - K.sum(K.square(anchor_pos), axis=-1))
#     negative_agrement_projection_loss = K.square(K.abs(K.batch_dot(negative, anchor_neg)) - K.sum(K.square(anchor_neg), axis=-1))

#     basic_loss = neg_dist + pos_dist + contrastive_projection_loss + positive_agrement_projection_loss + negative_agrement_projection_loss
#     loss = K.mean(basic_loss)
 
#     return loss

# def polarization_loss_v2(y_true, y_pred):
#     """
#     Implementation of the triplet loss function
#     Arguments:
#     y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor data
#             positive -- the encodings for the positive data (similar to anchor)
#             negative -- the encodings for the negative data (different from anchor)
#     Returns:
#     loss -- real number, value of the loss

#     my opinion:

#     y_pred: [0.8, 0.5, 0.3, 0.6, 0.4, 0.7]
#     y_true: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

#     dot_pred = square(dot_product(y_pred, [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    
#     """
#     return None

