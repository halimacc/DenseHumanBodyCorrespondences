"""This module provides the a softmax cross entropy loss for training FCN.
In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""
import tensorflow as tf

def softmax_cross_entropy(logits, labels):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-4)
        softmax = tf.nn.softmax(logits) + epsilon
        cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[-1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
