import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
from benchmark.eval_functions.resnet.resnet import inference
from benchmark.eval_functions.resnet.cifar10_input import random_crop_and_flip, whitening_image

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3


class Train(object):
    def __init__(self, train_dataset, test_dataset, num_classes, flags, seed=32, proportion=0.2):
        # Set up all the placeholders
        self.FLAGS = flags
        self.placeholders()
        self.train_dir = 'logs_' + self.FLAGS.version + '/'

        # Prepare dataset.
        self.test_dataset = test_dataset
        x_train, y_train = train_dataset
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train,
                                                                              test_size=proportion, random_state=seed)
        self.x_test = test_dataset[0]
        self.y_test = test_dataset[1]
        self.NUM_CLASS = num_classes
        self.TRAIN_SIZE = self.y_train.shape[0]
        self.VAL_SIZE = self.y_val.shape[0]
        self.TEST_SIZE = self.test_dataset[1].shape[0]

    def prepare_train_data(self):
        pad_width = ((0, 0), (self.FLAGS.padding_size, self.FLAGS.padding_size),
                     (self.FLAGS.padding_size, self.FLAGS.padding_size), (0, 0))
        padded_x_train = np.pad(self.x_train, pad_width=pad_width, mode='constant', constant_values=0)
        return padded_x_train, self.y_train

    def prepare_validation_data(self):
        validation_array = whitening_image(self.x_val)
        return validation_array, self.y_val

    def prepare_test_data(self):
        test_array = whitening_image(self.x_test)
        return test_array, self.y_test

    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.FLAGS.train_batch_size, IMG_HEIGHT,
                                                                         IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.FLAGS.train_batch_size])
        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.FLAGS.validation_batch_size])
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def build_train_validation_graph(self):
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, self.FLAGS.num_residual_blocks, reuse=False,
                           weight_decay=self.FLAGS.weight_decay, num_classes=self.NUM_CLASS)
        vali_logits = inference(self.vali_image_placeholder, self.FLAGS.num_residual_blocks, reuse=True,
                                weight_decay=self.FLAGS.weight_decay, num_classes=self.NUM_CLASS)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)

    def train(self):
        # For the first step, we are loading all training images and validation images into the
        # memory
        all_data, all_labels = self.prepare_train_data()
        vali_data, vali_labels = self.prepare_validation_data()

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # If you want to load from a checkpoint
            if self.FLAGS.is_use_ckpt is True:
                saver.restore(sess, self.FLAGS.ckpt_path)
                print('Restored from checkpoint...')
            else:
                sess.run(init)

            # This summary writer object helps write summaries on tensorboard
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

            # These lists are used to save a csv file at last
            step_list = []
            train_error_list = []
            val_error_list = []

            print('Start training...')
            print('----------------------------')

            for step in range(1, 1+self.FLAGS.train_steps):

                train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                            self.FLAGS.train_batch_size)

                validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                               vali_labels, self.FLAGS.validation_batch_size)
                # Want to validate once before training. You may check the theoretical validation
                # loss first
                if step % self.FLAGS.report_freq == 0:

                    if self.FLAGS.is_full_validation is True or (step == self.FLAGS.train_steps):
                        validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                                top1_error=self.vali_top1_error, vali_data=vali_data,
                                                vali_labels=vali_labels, session=sess,
                                                batch_data=train_batch_data, batch_label=train_batch_labels)

                        vali_summ = tf.Summary()
                        vali_summ.value.add(tag='full_validation_error',
                                            simple_value=validation_error_value.astype(np.float))
                        summary_writer.add_summary(vali_summ, step)
                        summary_writer.flush()

                    else:
                        _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                         self.vali_top1_error,
                                                                     self.vali_loss],
                                                    {self.image_placeholder: train_batch_data,
                                                     self.label_placeholder: train_batch_labels,
                                                     self.vali_image_placeholder: validation_batch_data,
                                                     self.vali_label_placeholder: validation_batch_labels,
                                                     self.lr_placeholder: self.FLAGS.init_lr})

                    val_error_list.append(validation_error_value)

                start_time = time.time()

                _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                                      self.full_loss, self.train_top1_error],
                                                                     {self.image_placeholder: train_batch_data,
                                                                      self.label_placeholder: train_batch_labels,
                                                                      self.vali_image_placeholder: validation_batch_data,
                                                                      self.vali_label_placeholder: validation_batch_labels,
                                                                      self.lr_placeholder: self.FLAGS.init_lr})
                duration = time.time() - start_time

                if step % self.FLAGS.report_freq == 0:
                    summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                        self.label_placeholder: train_batch_labels,
                                                        self.vali_image_placeholder: validation_batch_data,
                                                        self.vali_label_placeholder: validation_batch_labels,
                                                        self.lr_placeholder: self.FLAGS.init_lr})
                    summary_writer.add_summary(summary_str, step)

                    num_examples_per_step = self.FLAGS.train_batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec, sec_per_batch))
                    print('Train top1 error = ', train_error_value)
                    print('Validation top1 error = %.4f' % validation_error_value)
                    print('Validation loss = ', validation_loss_value)
                    print('----------------------------')

                    step_list.append(step)
                    train_error_list.append(train_error_value)

                if step == self.FLAGS.decay_step0 or step == self.FLAGS.decay_step1:
                    self.FLAGS.init_lr = self.FLAGS.lr_decay_factor * self.FLAGS.init_lr
                    print('Learning rate decayed to ', self.FLAGS.init_lr)

                # Save checkpoints every 10 epochs.
                if step % (10*self.FLAGS.report_freq) == 0 or step == self.FLAGS.train_steps:
                    if self.FLAGS.save_ckpt:
                        checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                        df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                        'validation_error': val_error_list})
                        df.to_csv(self.train_dir + self.FLAGS.version + '_error.csv')

                    if step == self.FLAGS.train_steps:
                        test_data, test_labels = self.prepare_test_data()
                        # Evaluate the model on the test dataset.
                        test_loss_value, test_error_value = self.full_validation(loss=self.vali_loss,
                                                                                 top1_error=self.vali_top1_error,
                                                                                 vali_data=test_data,
                                                                                 vali_labels=test_labels,
                                                                                 session=sess,
                                                                                 batch_data=train_batch_data,
                                                                                 batch_label=train_batch_labels,
                                                                                 mode='test')
                        print('Test performance is', test_loss_value, test_error_value)
                        return val_error_list[-1], test_error_value

    def evaluate(self):
        test_data, test_labels = self.prepare_test_data()
        num_batches = self.TEST_SIZE // self.FLAGS.test_batch_size
        order = np.random.choice(self.TEST_SIZE, num_batches * self.FLAGS.test_batch_size)
        vali_data_subset = test_data[order, ...]
        vali_labels_subset = test_labels[order]

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.FLAGS.test_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.test_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.FLAGS.test_batch_size])
        # Build the test graph
        logits = inference(self.test_image_placeholder, self.FLAGS.num_residual_blocks, reuse=False,
                           weight_decay=self.FLAGS.weight_decay, num_classes=self.NUM_CLASS)
        predictions = tf.nn.softmax(logits)
        test_top1_error = self.top_k_error(predictions, self.test_label_placeholder, 1)
        test_loss = self.loss(logits, self.test_label_placeholder)
        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # checkpoint_path = os.path.join(self.train_dir, 'model.ckpt-%d.meta' % self.FLAGS.train_steps)
        checkpoint_path = os.path.join(self.train_dir, 'model.ckpt-%d' % self.FLAGS.train_steps)
        saver.restore(sess, checkpoint_path)
        print('Model restored from ', checkpoint_path)

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * self.FLAGS.test_batch_size
            feed_dict = {self.test_image_placeholder: vali_data_subset[offset:offset + self.FLAGS.test_batch_size,
                                                      ...],
                         self.test_label_placeholder: vali_labels_subset[
                                                      offset:offset + self.FLAGS.test_batch_size]
                         }
            error_loss, top1_error_value = sess.run([test_loss, test_top1_error], feed_dict=feed_dict)
            loss_list.append(error_loss)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)

    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance
        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // self.FLAGS.test_batch_size
        remain_images = num_test_images % self.FLAGS.test_batch_size
        print ('%i test batches in total...' % num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.FLAGS.test_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, self.FLAGS.num_residual_blocks, reuse=False,
                           weight_decay=self.FLAGS.weight_decay, num_classes=self.NUM_CLASS)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, self.FLAGS.test_ckpt_path)
        print ('Model restored from ', self.FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print('%i batches finished!' % step)
            offset = step * self.FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+self.FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, self.FLAGS.num_residual_blocks, reuse=True,
                               weight_decay=self.FLAGS.weight_decay, num_classes=self.NUM_CLASS)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array

    # Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(self.VAL_SIZE - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch

    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(self.TRAIN_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=self.FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+self.FLAGS.train_batch_size]

        return batch_data, batch_label

    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(self.FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=self.FLAGS.momentum,
                                         use_nesterov=self.FLAGS.nesterov)

        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op

    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op

    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label, mode='val'):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        val_size = self.VAL_SIZE
        if mode != 'val':
            val_size = self.TEST_SIZE

        num_batches = val_size // self.FLAGS.validation_batch_size
        order = np.random.choice(val_size, num_batches * self.FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * self.FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+self.FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+self.FLAGS.validation_batch_size],
                self.lr_placeholder: self.FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)
