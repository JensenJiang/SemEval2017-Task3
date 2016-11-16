#longest 242
#example 19990
"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

tf.app.flags.DEFINE_string('directory', '/media/robin/sorry/lab_data/v3.2',
                           'Directory to download data files and write the '
                           'converted result')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

class Sentence(object):
    def __init__(self):
        self.word = []
t_sentence = Sentence();
All_example = []

def convert_to():
  for i in range(1):
    example = All_example[i]
  #for example in All_example:
    #senten = images[index].tostring()
    sentence = ''.join(example.word);
    print(sentence);
    print(example.label);
    print(example.question_len);
    print(example.question_len+example.ans_len);
    #print(sentence)
    tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(example.question_len),
        'Total_length': _int64_feature(example.question_len+example.ans_len),
        'label': _int64_feature(example.label),
        'Question_Answer': _bytes_feature(sentence)}))
    writer.write(tmp.SerializeToString())
  writer.close()


def main(argv):

  f = open("vec_res.txt", "r")  
  
  filename = os.path.join(FLAGS.directory, 'train' + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  count = 0;
  for line in f:
    line=line.strip('\n')
    #print(line)
    if(line=="ans"):
      #print line
      t_sentence.question_len = count;
      count = 0;
    elif(line =="bad"):
      #print line
      t_sentence.ans_len = count;
      count = 0;
      t_sentence.label = -1;
      while(len(t_sentence.word)<300):
        t_sentence.word.append("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
      tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(t_sentence.question_len),
        'Total_length': _int64_feature(t_sentence.question_len+t_sentence.ans_len),
        'label': _int64_feature(t_sentence.label),
        'Question_Answer': _bytes_feature(''.join(t_sentence.word))}))
      writer.write(tmp.SerializeToString())
      t_sentence.word = []
    elif(line =="good"):
      #print line
      t_sentence.ans_len = count;
      count = 0;
      t_sentence.label = 1;
      while(len(t_sentence.word)<300):
        t_sentence.word.append("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
      tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(t_sentence.question_len),
        'Total_length': _int64_feature(t_sentence.question_len+t_sentence.ans_len),
        'label': _int64_feature(t_sentence.label),
        'Question_Answer': _bytes_feature(''.join(t_sentence.word))}))
      writer.write(tmp.SerializeToString())
      t_sentence.word = []
    elif(line =="potentiallyuseful"):
      #print line
      t_sentence.ans_len = count;
      count = 0;
      t_sentence.label = 0;
      while(len(t_sentence.word)<300):
        t_sentence.word.append("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
      tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(t_sentence.question_len),
        'Total_length': _int64_feature(t_sentence.question_len+t_sentence.ans_len),
        'label': _int64_feature(t_sentence.label),
        'Question_Answer': _bytes_feature(''.join(t_sentence.word))}))
      writer.write(tmp.SerializeToString())
      t_sentence.word = []
    else:
      t_sentence.word.append(line)
      count = count + 1;

  f.close()  
  writer.close()
 # convert_to(t_sentence)

if __name__ == '__main__':
  tf.app.run()
