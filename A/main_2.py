import argparse
import os
import tensorflow as tf

# Args
parser = argparse.ArgumentParser()

## Training Parameters
parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('-it', '--iterate', type=int, help='Training iterates', default=1000)
parser.add_argument('-b', '--batch_size', type=int, help='Batch size', default=10)
parser.add_argument('-ds', '--display_step', type=int, help='Display step', default=5)
parser.add_argument('-wl', '--word_vec_length', type=int, help='Word vector length', default=25)

## NN Parameters
parser.add_argument('-sl', '--seq_len', type=int, help='Fixed length of sequences', default=300)
parser.add_argument('--hidden', type=int, help='Hidden layer number of features', default=64)

## IO Parameters
parser.add_argument('--dir', help='Dataset directory', default=r'')
parser.add_argument('-f', '--filename', help='Training dataset filename')
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('-t', '--num_threads', type=int, help='Number of threads', default=3)

args = parser.parse_args()

# IO Functions
def inputs(train_sets):
    if not args.epochs: args.epochs = None
    filename_queue = tf.train.string_input_producer(train_sets, num_epochs=args.epochs)
    return tf.train.shuffle_batch(
        read_and_decode(filename_queue),
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        capacity=1000 + 3 * args.batch_size,
        min_after_dequeue=1000,
        allow_smaller_final_batch=True,
    )

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'Question_length': tf.FixedLenFeature([], tf.int64),
            'Total_length': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([1], tf.int64),
            'Question_Answer': tf.FixedLenFeature([args.word_vec_length * args.seq_len], tf.float32),
        })
    # Some preprocess
    x = tf.reshape(features['Question_Answer'], [args.seq_len, args.word_vec_length])
    return x, features['label'], features['Total_length'], features['Question_length']

def biLSTM_AP(q, a, seqlen_q, seqlen_a, U):
    
    pass

if __name__ == '__main__':
    train_sets = [os.path.join(args.dir, args.filename)]
    x, y, seqlen_t, seqlen_q = inputs(train_sets)
    seqlen_a = tf.subtract(seqlent_t, seqlen_q)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            print(r'-------------------------------------------------------------------')
            print sess.run(y)
            #print(sess.run([x, y, seqlen_q, seqlen_t]))
    except tf.errors.OutOfRangeError:
        print('Exhaust!')
    finally:
        coord.request_stop()
