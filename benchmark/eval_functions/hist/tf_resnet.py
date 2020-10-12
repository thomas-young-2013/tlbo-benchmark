from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from official.resnet import resnet_model


# Get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


def resnet(features, reuse, is_training, num_classes):
    with tf.variable_scope('resnet_model', reuse=reuse):
        model = resnet_model.Cifar10Model(56, data_format='channels_last', num_classes=num_classes,
                             resnet_version=resnet_model.DEFAULT_VERSION, dtype=resnet_model.DEFAULT_DTYPE)
        logits = model(features, is_training)
    return logits


def train(num_classes, epoch_num, params, train_dataset, proportion=0.2, seed=32):
    import tensorflow as tf
    print(params)
    x_train, y_train = train_dataset
    x_train = x_train.astype('float32')
    x_train /= 255
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=proportion, random_state=seed)
    epoch_size = x_train.shape[0]
    # epoch_size = 10000
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    lr_reductions = params['lr_reductions']
    batch_size = int(params['batch_size'])
    nesterov = params['nesterov']

    boundary_epochs = [epoch_num//2, epoch_num*3//4, epoch_num]

    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels = tf.placeholder(tf.float32, [None, num_classes])
        logits = resnet(features, False, True, num_classes)
        logits_test = resnet(features, True, False, num_classes)

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(labels, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=nesterov)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # lr_value, epoch_index = learning_rate, 0
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init)
        # Training model.
        for step in range(epoch_size * epoch_num // batch_size):
            epoch_id = step*batch_size//epoch_size
            # if epoch_id == boundary_epochs[epoch_index]:
            #     lr_value *= lr_reductions
            #     epoch_index += 1
            #     print(epoch_id, 'current lr', lr_value)

            batch_x, batch_y = next_batch(batch_size, x_train, y_train)
            sess.run(train_op, feed_dict={features: batch_x, labels: batch_y})
            # if epoch_id%5 == 0 and step % epoch_size == 0:
            loss, acc = sess.run([loss_op, acc_op], feed_dict={features: batch_x, labels: batch_y})
            print(("Epoch %d" % epoch_id) + " loss: " + "{:.4f}".format(loss) + " acc: " + "{:.3f}".format(acc))

        # Evaluating model.
        def validate():
            test_idx = 0
            acc_list, loss_list = list(), list()
            while test_idx < x_val.shape[0]:
                test_x, test_y = x_val[test_idx: test_idx + batch_size], y_val[test_idx: test_idx + batch_size]
                loss, acc = sess.run([loss_op, acc_op], feed_dict={features: test_x, labels: test_y})
                acc_list.append(acc)
                loss_list.append(loss)
                test_idx += batch_size
            return np.mean(loss_list), np.mean(acc_list)
        _, acc = validate()
    return acc
