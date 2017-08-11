#! /usr/bin/python2.7

import cv2
import os
import math
import numpy as np
import tensorflow as tf
#import download
from tensorflow.contrib.tensorboard.plugins import projector



# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
#data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "/data/models/slim/classify_image_graph_def.pb"

# Directory to store the downloaded data.
data_dir = "/data/models/slim/"

#IMAGE_DIR = '/data/models/slim/retrainedmodel/prescan_jpg_embedding'
#LOG_DIR = '/data/models/slim/logs_aug11_7classes'
IMAGE_DIR = '/data/tmp/flowers_cifar_mnist/flowers/flower_photos/flower_pics'
LOG_DIR = '/data/models/slim/logs_flowers'


data_dir_list = os.listdir(IMAGE_DIR)

img_data = []
for dataset in data_dir_list:
    img_list = os.listdir(IMAGE_DIR+ '/' + dataset)
    print ('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(IMAGE_DIR + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (280, 280))
        img_data.append(input_img_resize)

img_data = np.array(img_data)



def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data



def load_graph():
    # Create a new TensorFlow computational graph.
    graph = tf.Graph()

    # Set the new graph as the default.
    with graph.as_default():
        # Open the graph-def file for binary reading.
        path = os.path.join(data_dir, path_graph_def)
        with tf.gfile.FastGFile(path, 'rb') as file:
            # The graph-def is a saved copy of a TensorFlow graph.
            # First we need to create an empty graph-def.
            graph_def = tf.GraphDef()

            # Then we load the proto-buf file into the graph-def.
            graph_def.ParseFromString(file.read())

            # Finally we import the graph-def to the default TensorFlow graph.
            tf.import_graph_def(graph_def, name='')
    return graph


def main(argv=None):
    #maybe_download()
    graph = load_graph()

    #basedir = os.path.dirname(__file__)

    # ensure log directory exists
    logs_path =  LOG_DIR
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
     #generate metadata
    with tf.Session(graph=graph) as sess:

        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        jpeg_data = tf.placeholder(tf.string)
        #thumbnail = tf.cast(tf.image.resize_images(tf.image.decode_jpeg(jpeg_data, channels=3), [100, 100]), tf.uint8)

        outputs = []
        images = []

        # Create metadata
        metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
        metadata = open(metadata_path, 'w')
        metadata.write("Name\tLabels\n")

        for folder_name in os.listdir(IMAGE_DIR):
            for file_name in os.listdir(IMAGE_DIR + '/' + folder_name):
                if not file_name.endswith('.jpg'):
                    continue
                print('process %s...' % file_name)

                with open(os.path.join( IMAGE_DIR + '/' + folder_name, file_name), 'rb') as f:
                    data = f.read()
                    results = sess.run(pool3, {
                        'DecodeJpeg/contents:0': data, jpeg_data: data})
                    #print("results", results)
                    outputs.append(results[0])
                    #images.append(results[1])
                    #print("images", outputs)
                    metadata.write('{}\t{}\n'.format(file_name, folder_name))
        metadata.close()

        embedding_var = tf.Variable(tf.stack(
            [tf.squeeze(x) for x in outputs], axis=0), trainable=False, name='embed')

        # prepare projector config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        summary_writer = tf.summary.FileWriter( LOG_DIR)

        # link metadata
        embedding.metadata_path = metadata_path

        # write to sprite image file
        img = images_to_sprite(img_data)
        cv2.imwrite(os.path.join(LOG_DIR, 'sprite_classes.png'), img)
        #print("img", img)

        embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_classes.png')
        embedding.sprite.single_image_dim.extend([100, 100])

        # save embedding_var
        projector.visualize_embeddings(summary_writer, config)
        sess.run(tf.variables_initializer([embedding_var]))

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
