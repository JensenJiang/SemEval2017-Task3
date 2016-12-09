import tensorflow as tf
from test_read_tfdata import inputs

# Parameters
learning_rate = 0.01
training_iters = 256
batch_size = 128
display_step = 100
word_vec_length = 25

# Network Parameters
seq_max_len = 300# handling variable-length sequences, we need to pad sequences to fixed-length
n_hidden = 64 # hidden layer number of features
n_classes = 3 # good, potentially useful and bad


def RNN(x, seqlen_t, seqlen_q, weights, biases):
    """ The main function for LSTM
        shape of x: (batch_size, seq_max_len, word_vec_length)
    """
    # Preprocess Data
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, word_vec_length])
    x = tf.split(0, seq_max_len, x)

    # Create Network
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden) # more parameters refer to the documentation
    outputs, states = tf.nn.rnn(lstm_cell, x,
                                dtype=tf.float32,
                                sequence_length=tf.reshape(seqlen_t,[-1])) 
    
    # Output Process
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2]) # change back to (batch_size, seq_max_len, n_hidden)
    
    batch_size = tf.shape(outputs)[0]
    index_a = tf.range(0, batch_size) * seq_max_len + (seqlen_t - 1) # generate index for valid response outputs
    index_q = tf.range(0, batch_size) * seq_max_len + (seqlen_q - 1)

    outputs = tf.reshape(outputs, [-1, n_hidden]) # is the shape right?
    outputs_q = tf.gather(outputs, index_q)
    outputs_a = tf.gather(outputs, index_a)
    temp1 = tf.matmul(outputs_q, weights['out'])
    temp2 = tf.multiply(temp1, outputs_a)

    return tf.sigmoid(tf.reduce_sum(temp2, 1) + biases['out'])

# Evaluate Model
#correct_pred =    # how to evaluate???
#accuracy = 

# Launch
with tf.Graph().as_default():
    # Graph Input
    x = tf.placeholder(tf.float32, [None, seq_max_len, word_vec_length],'hrb_x')
    y = tf.placeholder(tf.float32, [None,1],'hrb_y') # ???
    #seqlen_q = tf.placeholder(tf.int32, [None,1],'hrb-seqlen-q')
    seqlen_q = tf.placeholder(tf.int32)
    #seqlen_t = tf.placeholder(tf.int32, [None,1],'hrb-seqlen-t')
    seqlen_t = tf.placeholder(tf.int32)

    # Weights
    weights = {
       'out': tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([]))
    }

    pred = RNN(x, seqlen_t, seqlen_q, weights, biases)

    # Loss Function and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    q_batch_x, q_batch_y, q_batch_seqlen_q, q_batch_seqlen_t = inputs([r'./train.tfrecords'])
    print("Reading data finished")
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen_q, batch_seqlen_t = sess.run([q_batch_x, q_batch_y, q_batch_seqlen_q, q_batch_seqlen_t])
        feed_dict = {x: batch_x,
                     y: batch_y,
                     seqlen_q: batch_seqlen_q,
                     seqlen_t: batch_seqlen_t,}
        sess.run(optimizer, feed_dict=feed_dict)
        # Display
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            loss = sess.run(cost, feed_dict=feed_dict)

            print('Iter' + str(step*batch_size) + ', Minibatch Loss = ' + '{:.6f}'.format(loss) + ', Training Accuracy = ' + '{:.5f}'.format(acc))

        step += 1
    print('Finished!')

