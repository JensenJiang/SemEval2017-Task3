#longest 242
#example 19990
"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

tf.app.flags.DEFINE_string('directory', '/home/jensen/Workspace/Machine-Learning/SemEval2017/A',
                           'Directory to download data files and write the '
                           'converted result')

FLAGS = tf.app.flags.FLAGS
MAX_SEQ_LEN = 20
ZERO_VEC = [.0] * 25
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

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

def string_to_floatlist(word_str):
    """Return a list of float extracting from word_str"""
    strlist = word_str.split(' ')
    return [float(e) for e in strlist]

def main(argv):

  f = open("vec_res_test.txt", "r")  
  
  filename = os.path.join(FLAGS.directory, 'train' + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  
  count = 0;
  for line in f:
    line=line.strip('\n')
    line = line.strip(' ')
    #print(line)
    if(line=="ans"):
      #print lined
      t_sentence.question_len = count
      count = 0
    elif(line =="bad"):
      #print line
      t_sentence.ans_len = count
      count = 0
      t_sentence.label = -1
      t_sentence.word.extend(ZERO_VEC * (MAX_SEQ_LEN - (t_sentence.question_len + t_sentence.ans_len)))
      tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(t_sentence.question_len),
        'Total_length': _int64_feature(t_sentence.question_len+t_sentence.ans_len),
        'label': _int64_feature(t_sentence.label),
        'Question_Answer': _float_feature(t_sentence.word)}))
      writer.write(tmp.SerializeToString())
      t_sentence.word = []
    elif(line =="good"):
      #print line
      t_sentence.ans_len = count;
      count = 0;
      t_sentence.label = 1;
      t_sentence.word.extend(ZERO_VEC * (MAX_SEQ_LEN - (t_sentence.question_len + t_sentence.ans_len)))
      tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(t_sentence.question_len),
        'Total_length': _int64_feature(t_sentence.question_len+t_sentence.ans_len),
        'label': _int64_feature(t_sentence.label),
        'Question_Answer': _float_feature(t_sentence.word)}))
      writer.write(tmp.SerializeToString())
      t_sentence.word = []
    elif(line =="potentiallyuseful"):
      #print line
      t_sentence.ans_len = count;
      count = 0;
      t_sentence.label = 0;
      t_sentence.word.extend(ZERO_VEC * (MAX_SEQ_LEN - (t_sentence.question_len + t_sentence.ans_len)))
      tmp = tf.train.Example(features=tf.train.Features(feature={
        'Question_length': _int64_feature(t_sentence.question_len),
        'Total_length': _int64_feature(t_sentence.question_len+t_sentence.ans_len),
        'label': _int64_feature(t_sentence.label),
        'Question_Answer': _float_feature(t_sentence.word)}))
      writer.write(tmp.SerializeToString())
      t_sentence.word = []
    else:
      t_sentence.word.extend(string_to_floatlist(line))
      count = count + 1

  f.close()  
  writer.close()
 # convert_to(t_sentence)

if __name__ == '__main__':
  tf.app.run()
