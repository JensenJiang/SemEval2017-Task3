import tensorflow as tf

# Parameters
learning_rate = 0.01
training_iters = 10000
batch_size = 128
display_step = 100
word_vec_length = 25

# Network Parameters
seq_max_len = 20# handling variable-length sequences, we need to pad sequences to fixed-length
n_hidden = 64 # hidden layer number of features
n_classes = 3 # good, potentially useful and bad

# Graph Input
x = tf.placeholder(tf.float32, [None, seq_max_len, word_vec_length])
y = tf.placeholder(tf.float32, [None]) # ???
seqlen_q = tf.placeholder(tf.int32, [None])
seqlen_a = tf.placeholder(tf.int32, [None])

# Weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_hidden]))
}
biases = {
    'out': tf.Variable(tf.random_normal([]))
}

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
                                sequence_length=seqlen_t) 
    
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
    
    list_a1 = tf.split(0, batch_size, temp1)
    list_a2 = tf.split(0, batch_size, outputs_a)
    dot_pro = tf.pack([tf.reduce_sum(tf.mul(list_a1[i] * list_a2[i])) for i in range(0, batch_size)])

    return tf.sigmoid(dot_pro + biases)

pred = RNN(x, seqlen_q + seqlen_a, seqlen_q, weights, biases)

# Loss Function and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate Model
#correct_pred =    # how to evaluate???
#accuracy = 

# Initializing the variables
init = tf.initialize_all_variables()

# Launch
with tf.Graph().as_default():
    sess = tf.Session()
    sess.run(init)
    coord = tf.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_seqlen_q, batch_seqlen_t, batch_y = inputs([r'./train.tfrecords'])
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

