from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_string('train_dir', '/media/robin/sorry/lab_data/v3.2',
                           'Directory to download data files and write the '
                           'converted result')
FLAGS = flags.FLAGS

TRAIN_FILE = './train.tfrecords'

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'Question_length':tf.FixedLenFeature([1], tf.int64),
          'Total_length':tf.FixedLenFeature([1], tf.int64),
          'label':tf.FixedLenFeature([1], tf.int64),
          'Question_Answer':tf.FixedLenFeature([500], tf.float32),
      })
  x = tf.cast(features['Question_Answer'], tf.float32)
  x = tf.reshape(x, [20,25])
  y = tf.cast(features['label'], tf.int64)
  seqlen_t = tf.cast(features['Total_length'], tf.int64)
  seqlen_q = tf.cast(features['Question_length'], tf.int64)
  return x, y, seqlen_q, seqlen_t


def inputs():
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([TRAIN_FILE])
        _x, _y, _seqlen_q, _seqlen_t = read_and_decode(filename_queue)
        x, y, seqlen_q, seqlen_t = tf.train.shuffle_batch(
            [_x, _y, _seqlen_q, _seqlen_t], batch_size=2,
            num_threads=1,
            capacity=5,
            min_after_dequeue=3,
            allow_smaller_final_batch=True)
        return x, y, seqlen_q, seqlen_t

if __name__ == '__main__':
    with tf.Graph().as_default(): 
        x, y, seqlen_q, seqlen_t = inputs()
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            print(r'-------------------------------------------------------------------')
            print(sess.run([x, y, seqlen_q, seqlen_t]))

