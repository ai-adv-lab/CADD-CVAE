"""
The MIT License

 

Copyright (c) 2020 Samsung SDS

 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import os
import glob
import datetime

import argparse

import cvae
import data_utils

NUM_LABELS = 17
NUM_TO_GENERATE = NUM_LABELS * 500


def parse_args():
    parser.add_argument('--results_path', type=str, default='results_cvae',
                        help='File path of output images')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent space', required=True)
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run', required=True)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --results_path
    try:
        folder_name = "/{0}_{1}".format(str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")).replace(' ', ''),
                                        'CADD_CVAE')
        args.results_path = args.results_path + '/' + folder_name
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path + '/*')
    for f in files:
        os.remove(f)

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learning_rate
    try:
        assert args.learning_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main function"""


def main(args):
    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture
    n_hidden = args.n_hidden
    dim_z = args.dim_z
    num_bits = 166  # 166 bit MACCS key

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # onehot_label = args.target_label+5

    data_list = data_utils.get_input_gi50_data("./fp.loggi50.onehot.npy", test_div=10, normalize=False, label = NUM_LABELS )
    test_data, train_total_data, test_labels, train_label = data_list
    n_samples = train_total_data.shape[0]
    print('info: test_labels.shape -> {}'.format(train_label.shape))
    np.save('data/test_data.npy',test_data)

    """ build graph """
    with open(RESULTS_DIR + '/log.txt', 'a') as dec_fp:
        dec_fp.write('input x dimension : %d \n' % num_bits)
        dec_fp.write('input y label dimension : %d \n' % NUM_LABELS)
        dec_fp.write('2,4,6,8,10 <= -logGI50 < 3,5,7,9,11 \n')

        dec_fp.write('encoder layer [%d, %d] \n' % (n_hidden, n_hidden))
        dec_fp.write('encoder activation function [elu, tanh] \n')
        dec_fp.write('encoder dropout keep_prob [0.9, 0.9] \n')
        dec_fp.write('latent space z : %d \n' % dim_z)
        dec_fp.write('\n')
        dec_fp.write('decoder layer [%d, %d] \n' % (n_hidden, n_hidden))
        dec_fp.write('decoder activation function [tanh, elu] \n')
        dec_fp.write('decoder dropout keep_prob [0.9, 0.9] \n')
        dec_fp.write('output activation function [sigmoid] \n')
        dec_fp.write('\n')
        dec_fp.write('re-parameterization N(0,1) \n')
        dec_fp.write('batch size : %d \n' % batch_size)
        dec_fp.write('learning rate : %d \n' % learning_rate)

    # input placeholders
    x_hat = tf.placeholder(tf.float32, shape=[None, num_bits], name='input_fp')
    x = tf.placeholder(tf.float32, shape=[None, num_bits], name='target_fp')
    y = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name='target_labels')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for fp generation
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
    condition_in = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name='conditional_vectors')  # condition

    # network architecture
    x_, z, loss, neg_marginal_likelihood, KL_divergence = cvae.autoencoder(x_hat, x, y, num_bits, dim_z, n_hidden,
                                                                           keep_prob)
    # optimization
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    """ training """

    decoded = cvae.decoder(z_in, condition_in, num_bits, n_hidden)

    # train
    total_batch = int(n_samples / batch_size)
    print("n_samples : ", str(n_samples))
    min_tot_loss = 1e99

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-NUM_LABELS]
            # 0-1 to 0.001 - 0.999
            train_data_[train_data_[:] == 0] = 0.001
            train_data_[train_data_[:] == 1] = 0.999

            train_labels_ = train_total_data[:, -NUM_LABELS:]

            # Loop over all batches
            print("total_batch : ", str(total_batch))
            for i in range(total_batch):
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_ys_input = train_labels_[offset:(offset + batch_size)]
                batch_xs_target = batch_xs_input

                # add salt & pepper noise
                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, y: batch_ys_input, keep_prob: 0.9})

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))
            with open(RESULTS_DIR + '/log.txt', 'a') as epoch_log:
                epoch_log.write("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f \n" % (
                    epoch, tot_loss, loss_likelihood, loss_divergence))

            # if minimum loss is updated or final epoch, make results
            if min_tot_loss > tot_loss or epoch + 1 == n_epochs:
                min_tot_loss = tot_loss

                # Plot for analogical reasoning result
                if epoch > 600:
                    # onehot_label= [0:-3<<-2, 1:-2<<-1, 2:-1<<0, 3:0<<1, 4:1<<2, 5:2<<3, 6:3<<4, 7:4<<5, 8:5<<6, 9:6<<7, 10:7<<8, 11:8<<9, 12:9<<10, 13:10<<11, 14:11<<12, 15:12<<13, 16: 13<<14]

                    z = np.random.randn(NUM_TO_GENERATE, dim_z) * 1.
                    conditions_to_generate = np.zeros(shape=[NUM_TO_GENERATE, NUM_LABELS])
                    for i in range(NUM_TO_GENERATE):
                        label = i % NUM_LABELS
                        conditions_to_generate[i, label] = 1.0
                    x_output = sess.run(decoded, feed_dict={z_in: z,
                                                            condition_in: conditions_to_generate,
                                                            keep_prob: 1})
                    print('info: x_vector.shape -> {}'.format(x_output.shape))

                    x_output_bit = np.where(x_output > 0.5, 1, 0)

                    """ generate fingerprint """
                    # onehot_label = args.target_label+5
                    condition_gi = [4, 5, 6, 7, 8]
                    for i in condition_gi:
                        dec_fp = open(RESULTS_DIR + '/dec_fp_v' + str(i) + '_e%d.txt' % epoch, 'a')
                        for j, fp in enumerate(x_output_bit.tolist()):
                            fp_str = ""
                            if j % 18 == i + 3 + 1:
                                for bit in fp:
                                    fp_str += str(bit)
                                dec_fp.write(fp_str + ',%f' % i)
                                dec_fp.write('\n')
                        dec_fp.close()


if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
