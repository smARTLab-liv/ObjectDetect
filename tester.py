import tensorflow as tf
import numpy as np

with tf.Session() as sess:

    Num = 3
    Len = 4

    # input
    input = tf.placeholder(tf.float32, shape = (Num, Len))

    # normalize each row
    normalized = tf.nn.l2_normalize(input, dim = 1)

    # multiply row i with row j using transpose
    # element wise product
    prod = tf.matmul(input, input, adjoint_b = True)  # transpose second matrix

    input_matrix = np.array(
        [[ 1, 0, 0, 0 ],
         [ 0, 0, 1, 0 ],
         [ 1, 0, 0, 0 ]
         ],
        dtype = 'float32')

    print "input_matrix:"
    print input_matrix

    print "tensorflow:"
    print sess.run(prod, feed_dict = { input : input_matrix })
