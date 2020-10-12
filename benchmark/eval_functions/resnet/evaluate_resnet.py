import tensorflow as tf
from benchmark.eval_functions.resnet.cifar10_train import Train


def create_flags(params, epoch_num, epoch_size):
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    lr_decay_factor = params['lr_decay_factor']
    batch_size = int(params['batch_size'])
    nesterov = params['nesterov']
    weight_decay = params['weight_decay']
    padding_size = int(params['padding_size'])

    train_steps = epoch_size*epoch_num // batch_size
    report_freq = 2 * epoch_size // batch_size
    if epoch_num >= 150:
        report_freq = 10 * epoch_size // batch_size
    FLAGS = tf.app.flags.FLAGS

    # The following flags are related to save paths, tensorboard outputs and screen outputs
    tf.app.flags.DEFINE_string('version', 'test_110', '''A version number defining the directory to save
    logs and checkpoints''')
    tf.app.flags.DEFINE_integer('report_freq', report_freq, '''Steps takes to output errors on the screen
    and write summaries''')
    tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
    moving average shown on tensorboard''')

    # The following flags define hyper-parameters regards training
    tf.app.flags.DEFINE_integer('train_steps', train_steps, '''Total steps that you want to train''')
    tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
    a random batch''')
    tf.app.flags.DEFINE_integer('train_batch_size', batch_size, '''Train batch size''')
    tf.app.flags.DEFINE_integer('validation_batch_size', 125, '''Validation batch size, better to be
    a divisor of 10000 for this task''')
    tf.app.flags.DEFINE_integer('test_batch_size', 125, '''Test batch size''')

    tf.app.flags.DEFINE_float('init_lr', learning_rate, '''Initial learning rate''')
    tf.app.flags.DEFINE_float('lr_decay_factor', lr_decay_factor, '''How much to decay the learning rate each
    time''')
    tf.app.flags.DEFINE_integer('decay_step0', train_steps//2, '''At which step to decay the learning rate''')
    tf.app.flags.DEFINE_integer('decay_step1', (train_steps*3)//4, '''At which step to decay the learning rate''')

    # The following flags define hyper-parameters modifying the training network
    tf.app.flags.DEFINE_integer('num_residual_blocks', 8, '''How many residual blocks do you want''')
    tf.app.flags.DEFINE_float('weight_decay', weight_decay, '''scale for l2 regularization''')
    tf.app.flags.DEFINE_float('momentum', momentum, '''value for momentum''')
    tf.app.flags.DEFINE_boolean('nesterov', nesterov, '''use nesterov or not''')

    # The following flags are related to data-augmentation
    tf.app.flags.DEFINE_integer('padding_size', padding_size, '''In data augmentation, layers of zero padding on
    each side of the image''')

    # If you want to load a checkpoint and continue training
    tf.app.flags.DEFINE_string('ckpt_path', 'logs_test_110/model.ckpt-%d' % train_steps, '''Checkpoint
    directory to restore''')
    tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
    training''')
    tf.app.flags.DEFINE_boolean('save_ckpt', False, '''Whether to save a checkpoint''')

    tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999', '''Checkpoint
    directory to restore''')
    return FLAGS


def train(num_classes, epoch_num, params, train_dataset, test_dataset, proportion=0.2, seed=32):
    import tensorflow as tf
    # Construct the configuration flags.
    epoch_size = int(train_dataset[1].shape[0]*(1-proportion))
    flags = create_flags(params, epoch_num, epoch_size)
    train = Train(train_dataset, test_dataset, num_classes, flags, seed=seed, proportion=proportion)
    # Train the resnet models.
    val_error, test_error = train.train()
    return float(val_error), float(test_error)
