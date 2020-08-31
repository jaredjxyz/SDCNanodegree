import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    wanted_tensors = ('image_input:0', 'keep_prob:0', 'layer3_out:0', 'layer4_out:0', 'layer7_out:0')
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    special_tensors = tuple(sess.graph.get_tensor_by_name(tensor_name) for tensor_name in wanted_tensors)
    op = sess.graph.get_operations()
    
    return special_tensors

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    upsampled_1 = tf.layers.conv2d_transpose(conv1x1_7, num_classes, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip_1 = tf.add(upsampled_1, conv1x1_4)

    upsampled_2 = tf.layers.conv2d_transpose(skip_1, num_classes, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip_2 = tf.add(upsampled_2, conv1x1_3)

    upsampled_3 = tf.layers.conv2d_transpose(skip_2, num_classes, kernel_size=16, strides=(8,8), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return upsampled_3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return (logits, train_op, cross_entropy_loss)
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :par`am train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        batch_number = 0
        for input_images, output_images in get_batches_fn(batch_size):
            batch_number += 1
            sess.run(train_op, feed_dict={input_image: input_images, correct_label: output_images, keep_prob: .5})
            print(batch_number)
            print(sess.run(cross_entropy_loss, feed_dict={input_image: input_images, correct_label: output_images, keep_prob:.5}))



    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        epochs = 10
        batch_size=2
        learning_rate=.001

        labels = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], num_classes), name='labels')
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build neural network
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, 'vgg')
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, labels, learning_rate, num_classes)

        #Train NN using the train_nn function
        
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, labels, keep_prob, learning_rate)

        #Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
