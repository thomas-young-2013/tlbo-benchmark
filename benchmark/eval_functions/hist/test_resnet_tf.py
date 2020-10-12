from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

from keras.utils import to_categorical
from tensorflow.contrib.eager.python.examples.resnet50.resnet50 import ResNet50
from sklearn.model_selection import train_test_split

num_classes = 12
x_train = np.load('../../data/plant_train_img_224.npy')
y_train = np.load('../../data/plant_train_label_224.npy')
y_train = to_categorical(y_train, num_classes)
x_train = x_train.astype('float32')
x_train /= 255
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# Use validation set as evaluation target
x_test, y_test = x_val, y_val
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')
epoch_size = x_train.shape[0]


# Get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


def image_shape(batch_size):
    return [batch_size, 224, 224, 3]


def train_model(cls_num, img_size, epoch_num, params, train, proportion, seed, is_categorical=False):
    import tensorflow as tf
    epochs = 50
    batch_size = 32
    acc_t = 0.

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, image_shape(None))
        y = tf.placeholder(tf.float32, [None, num_classes])
        model = ResNet50('channels_last', include_top=False)

        inner_layer = model(X, training=True)
        logits = tf.layers.dense(inner_layer, num_classes)
        logits = tf.reshape(logits, [-1, num_classes])
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in range(epoch_size*epochs//batch_size):
                batch_x, batch_y = next_batch(batch_size, x_train, y_train)
                sess.run(train_op, feed_dict={X: batch_x, y: batch_y})
                loss_t, acc_t = sess.run([loss, acc_op], feed_dict={X: batch_x, y: batch_y})
                print(("Step %d" % step) + " loss: " + "{:.4f}".format(loss_t) + " acc: " + "{:.3f}".format(acc_t))
        return acc_t
