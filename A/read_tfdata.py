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

TRAIN_FILE = 'trains.tfrecords'

def read_and_decode(filename_queue):
  print("read and decode")
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'Question_length':tf.FixedLenFeature([], tf.int64),
          'Total_length':tf.FixedLenFeature([], tf.int64),
          'label':tf.FixedLenFeature([], tf.int64),
         'Question_Answer':tf.FixedLenFeature([], tf.string),
      })
  image = tf.decode_raw(features['Question_Answer'], tf.float32)
  label = tf.cast(features['label'], tf.int32)
  #print(label.eval())
  return image, label


def inputs():
  print("inputs")
  filename = os.path.join(FLAGS.train_dir,TRAIN_FILE)
  filename_queue = tf.train.string_input_producer([filename])
  image, sparse_labels = read_and_decode(filename_queue)
  return image, sparse_labels

def main(_):
  print("main")
  sess = tf.Session()

  filename = os.path.join(FLAGS.train_dir,TRAIN_FILE)
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'Question_length':tf.FixedLenFeature([], tf.int64),
          'Total_length':tf.FixedLenFeature([], tf.int64),
          'label':tf.FixedLenFeature([], tf.int64),
         'Question_Answer':tf.FixedLenFeature([], tf.string),
      })
  image = tf.decode_raw(features['Question_Answer'], tf.float32)
  labels = tf.cast(features['label'], tf.int32)
  #result = sess.run(labels)
  print(image)
  sess.close()

if __name__ == '__main__':
  tf.app.run()

