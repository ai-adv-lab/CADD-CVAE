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


"""
    data_utils.py

    CVAE for utility functions for drug discovery on oncology

    modified by jaeho3.yang@samsung.com (2018/03/16)

    requirements:
        python 3.x
        tensorflow 1.6+
"""
import numpy as np
import tensorflow as tf
import datetime
import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from pylab import figure, axes, scatter, title, show

def from_results(results_path, name):
    folder_name = "/{0}_{1}".format(str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")).replace(' ',''), name)
    #datetime.datetime.now()
    tensorboard_path = results_path + folder_name + "/Tensorboard"
    saved_model_path = results_path + folder_name + "/Saved_models"
    saved_image_path = results_path + folder_name + "/Images"
    log_path = results_path + folder_name + "/log"
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(saved_image_path)
        os.mkdir(log_path)

    return tensorboard_path, saved_model_path, saved_image_path, log_path
   
def get_input_data(data_file_name, unique_data_file_name, test_div=10, normalize=False):
    """ input data format assumption
        each line : 166 bits fingerprint, 1 concentration float32, 1 GI float32
    """
    data = np.load(data_file_name)
    # 166 bits fingerprint, 1 concentration float, 1 TGI float

    print('info: get_input_data: data.shape -> {}'.format(data.shape))


    unique_fp = np.load(unique_data_file_name)
    # there is 6252 unique fingerprints (and multiple experiments with each...)

    print('info: get_input_data: unique_fp.shape -> {}'.format(unique_fp.shape))

    """ 1. make randomize input data 
        2. split test_data, train_data
    """
    np.random.shuffle(data)
    test_size = data.shape[0] // test_div

    label_data = np.copy(data[:,-1:])

    #""" Normalize """
    #data[:,-1:] = data[:,-1:] * 0.01

    if normalize:
        data[:,-2:-1] = (data[:,-2:-1] - data[:,-2:-1].min(axis=0)) / \
                        (data[:,-2:-1].max(axis=0) - data[:,-2:-1].min(axis=0))
        data[:,-1:] = (data[:,-1:] - data[:,-1:].min(axis=0)) / (data[:,-1:].max(axis=0) - data[:,-1:].min(axis=0))

    test_data, train_data = np.vsplit(data, [test_size])
    test_label, train_label = np.vsplit(label_data, [test_size])

    return test_data, train_data, test_label, train_label

def get_input_gi50_data(gi50_data_file_name, test_div=10, normalize=False, label=17):
    """ input data format assumption
        each line : 166 bits fingerprint, 1 concentration float32, 1 GI float32
    """
    if gi50_data_file_name.endswith('.npz'):
            data1 = np.load(gi50_data_file_name)
            data = data1['arr_0']
    else :
        data = np.load(gi50_data_file_name)
    # 166 bits fingerprint, 1 log(GI50) float
    
    print('info: get_input_data: (gi50) data.shape -> {}'.format(data.shape))

    """ 1. make randomize input data 
        2. split test_data (first 100 of them), train_data (rest of them)
    """
    np.random.shuffle(data)
    test_size = data.shape[0] // test_div

    label_data = np.copy(data[:,-label:])

    """ Normalize """
    #data[:,-1:] = data[:,-1:] * 0.1

    if normalize:
        data[:,-1:] = (data[:,-1:] - data[:,-1:].min(axis=0)) / (data[:,-1:].max(axis=0) - data[:,-1:].min(axis=0))

    test_data, train_data = np.vsplit(data, [test_size])
    test_label, train_label = np.vsplit(label_data, [test_size])

    return test_data, train_data, test_label, train_label

def batch_gen(data, batch_size=64):
    """ python2: /
        python3: // (floordiv)
    """
    max_index = data.shape[0] // batch_size

    while True:
        np.random.shuffle(data)
        for i in range(max_index):
            yield np.hsplit(data[batch_size*i:batch_size*(i+1)], [-2, -1])

def batch_gen2(data, batch_size=64, input_space=166):
    """ python2: /
        python3: // (floordiv)
    """
    max_index = data.shape[0] // batch_size

    while True:
        np.random.shuffle(data)
        for i in range(max_index):
            yield np.hsplit(data[batch_size*i:batch_size*(i+1)], [input_space])

def same_gen(unq_fp, n_examples=64, n_different=1, mu=-5.82, std=1.68):
    """ Generator of same fingerprints with different concentration
       - called at pre_train(), train()
       :unq_fp: = maybe unique fingerprints (only) file (166 bits)
       :n_examples: =64 (fixed, not used as param)
       :n_different: =1 (32 used for param of pretrain(), train())
       :mu, std: (fixed, not used as param)
    """
    
    if n_examples % n_different: 
        raise ValueError('n_examples(%s) must be divisible by n_different(%s)' % (n_examples, n_different))
    max_index = unq_fp.shape[0] // n_different
    targets = np.zeros((n_examples, n_examples))
    block_size = n_examples // n_different
    for i in range(n_different):
        '''blocks of ones for every block of equal fp's'''
        targets[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = 1.
    targets = targets > 0
    
    while 1:
        np.random.shuffle(unq_fp)
        for i in range(max_index):
            batch_conc = np.random.normal(mu, std, size=(n_examples, 1))
            batch_fp = np.repeat(unq_fp[i*n_different:(i+1)*n_different], [block_size]*n_different, axis=0)
            yield batch_fp, batch_conc, targets
            
            
def uniform_initializer(size_1, size_2):
    normalized_size = np.sqrt(6) / (np.sqrt(size_1 + size_2))
    return tf.random_uniform([size_1, size_2], minval=-normalized_size, maxval=normalized_size)

def gauss_initializer(size_1, size_2):
    return tf.random_normal([size_1, size_2], 0, 2. / (size_1 * size_2))

def identity_function(x, name):
    return x

def get_collections_from_scope(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

def dense_layer(size_1, size_2, layer_input, layer_name,
                activation_function=tf.nn.relu,
                initialization_function=uniform_initializer,
                batch_normed=False, epsilon = 1e-3, ignore_b=False):
    """
    :size_1: current_layer size
    :size_2: next_layer size
    :layer_input: input tensor
    :layer_name: current layer name
    :activation_function: default is relu (use, leaky_relu)
    :initialization_function(s1, s2) : default is uniform_initializer
    :batch_normed: True / False
    :epsilon: (error) : 1e-3
    """
    print("dense_layer(name = {}, size_1 = {}, size_2 = {}, layer_input = {}".format(layer_name, size_1, size_2, layer_input.shape))
    w = tf.Variable(initialization_function(size_1, size_2), name="W")
    if not batch_normed:
        b = tf.Variable(tf.random_normal([size_2]), name="b")
        if ignore_b:
            return activation_function(tf.matmul(layer_input, w), name=layer_name)        
        else:
            return activation_function(tf.add(tf.matmul(layer_input, w), b), name=layer_name)

    pre_output = tf.matmul(layer_input, w)
    batch_mean, batch_var = tf.nn.moments(pre_output,[0])
    scale = tf.Variable(tf.ones([size_2]))
    beta = tf.Variable(tf.zeros([size_2]))
    layer_output_value = tf.nn.batch_normalization(pre_output, batch_mean, batch_var, beta, scale, epsilon)
    return activation_function(layer_output_value, name=layer_name)

def save_scattered_image(z, id, name='scattered_image.jpg'):
    N = 17
    plotn=2
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=id, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-4.5*plotn, 4.5*plotn])
    axes.set_ylim([-4.5*plotn, 4.5*plotn])
    plt.grid(True)
    plt.savefig(name)

def train_step(loss, scope_list, use_optimizer='adam', learning_rate=.001, momentum=.9):
    """
    :loss:
    :scope_list:
    :use_optimizer:
    :learning_rate:
    :momentum:

    :Raises: NotImplementedError: If an unsupported optimizer is requested.
    """

    if use_optimizer == 'adagrad':
        train_op = tf.train.AdagradOptimizer(learning_rate)
    elif use_optimizer == 'adagradDA':
        train_op = tf.train.AdagradDAOptimizer(learning_rate,
                                               global_step=tf.Variable(0, trainable=False, dtype='int64'))
    elif use_optimizer == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate,beta1=0.9)

    elif use_optimizer == 'momentum':
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum)
    elif use_optimizer == 'rmsprop':
        train_op = tf.train.RMSPropOptimizer(learning_rate, momentum)
    elif use_optimizer == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise NotImplementedError('Unsupported optimizer %s' % use_optimizer)

    variable_list = []
    for x in scope_list:
        variable_list += get_collections_from_scope(x)

    return train_op.minimize(loss, var_list=variable_list)


def get_variable_list(scope_list):
    variable_list = []
    for x in scope_list:
        variable_list += get_collections_from_scope(x)

    return variable_list


def show_image(input_x, latent_z, output_y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        ix = np.pad(input_x[i], ((0, 0), (0, 3)), 'constant', constant_values=0)
        # print('ix = {} {}'.format(ix.shape, ix[0].dtype))
        # print('   = {}'.format(ix[0:5]))
        plt.imshow(ix.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        lx = np.array(latent_z[i])
        # print('lx = {} {}'.format(lx.shape, lx[0].dtype))
        # print('   = {}'.format(lx[0:5]))
        plt.stem(lx.reshape(-1))
        plt.gray()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

        ax = plt.subplot(3, n, i + 1 + n + n)
        oy = np.pad(output_y[i], [(0, 0), (0, 3)], 'constant', constant_values=0)
        # print('oy = {} {}'.format(oy.shape, oy[0].dtype))
        # print('   = {}'.format(oy[0:5]))
        plt.imshow(oy.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save_flag:
        plt.savefig(save_dir + '/eval_e%d.png'%epoch_tag)
    else:
        plt.show()
    plt.close()

def show_image_sup(input_x, latent_z, output_y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n/5, i%10+1)
        ix = np.pad(input_x[i], ((0, 0), (0, 3)), 'constant', constant_values=0)
        # print('ix = {} {}'.format(ix.shape, ix[0].dtype))
        # print('   = {}'.format(ix[0:5]))
        plt.imshow(ix.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        lx = np.array(latent_z[i])
        # print('lx = {} {}'.format(lx.shape, lx[0].dtype))
        # print('   = {}'.format(lx[0:5]))
        plt.stem(lx.reshape(-1))
        plt.gray()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

        ax = plt.subplot(3, n, i + 1 + n + n)
        oy = np.pad(output_y[i], [(0, 0), (0, 3)], 'constant', constant_values=0)
        # print('oy = {} {}'.format(oy.shape, oy[0].dtype))
        # print('   = {}'.format(oy[0:5]))
        plt.imshow(oy.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save_flag:
        plt.savefig(save_dir + '/eval_sup_e%d.png'%epoch_tag)
    else:
        plt.show()
    plt.close()

def show_image_v2(input_x, conv_g, latent_z, conc_z, output_y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        ix = np.pad(input_x[i], ((0, 0), (0, 3)), 'constant', constant_values=0)
        gx = conv_g[i]
        ax.set_title(gx[0])
        # print('ix = {} {}'.format(ix.shape, ix[0].dtype))
        # print('   = {}'.format(ix[0:5]))
        plt.imshow(ix.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        cx = conc_z[i]
        ax.set_title(cx.reshape(-1))
        lx = np.array(latent_z[i])
        # print('lx = {} {}'.format(lx.shape, lx[0].dtype))
        # print('   = {}'.format(lx[0:5]))
        plt.stem(lx.reshape(-1))
        #plt.text(-0.5, 0, conc_v[i], ha='center', va='center', size=8)
        plt.gray()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

        ax = plt.subplot(3, n, i + 1 + n + n)
        oy = np.pad(output_y[i], [(0, 0), (0, 3)], 'constant', constant_values=0)
        # print('oy = {} {}'.format(oy.shape, oy[0].dtype))
        # print('   = {}'.format(oy[0:5]))
        plt.imshow(oy.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save_flag:
        plt.savefig(save_dir + '/eval_e%d.png'%epoch_tag)
    else:
        plt.show()
    plt.close()

def show_image_v3(X, Z, Y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        ix = np.pad(X['fp'][i], ((0, 0), (0, 3)), 'constant', constant_values=0)
        if 'conc' in X:
            gx = X['conc'][i][0].reshape(-1)
            ax.set_title(gx)
            #if i==0:
            #    print("x[conc] = {}".format(gx))

        # print('ix = {} {}'.format(ix.shape, ix[0].dtype))
        # print('   = {}'.format(ix[0:5]))
        plt.imshow(ix.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        if 'conc' in Z:
            cx = Z['conc'][i]
            ax.set_title(cx.reshape(-1))
            #if i==0:
            #    print("c[conc] = {}".format(cx))

        lx = np.array(Z['latent'][i])
        # print('lx = {} {}'.format(lx.shape, lx[0].dtype))
        # print('   = {}'.format(lx[0:5]))
        plt.stem(lx.reshape(-1))
        #plt.text(-0.5, 0, conc_v[i], ha='center', va='center', size=8)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)

        ax = plt.subplot(3, n, i + 1 + n + n)
        if 'conc' in Y:
            dx = Y['conc'][i]
            ax.set_title(dx.reshape(-1))
            #if i==0:
            #    print("d[conc] = {}".format(dx))

        oy = np.pad(Y['fp'][i], [(0, 0), (0, 3)], 'constant', constant_values=0)
        # print('oy = {} {}'.format(oy.shape, oy[0].dtype))
        # print('   = {}'.format(oy[0:5]))
        plt.imshow(oy.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save_flag:
        plt.savefig(save_dir + '/eval_e%d.png'%epoch_tag)
    else:
        plt.show()

def show_image_v4(X, Z, Y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        ix = np.pad(X['fp'][i], ((0, 0), (0, 3)), 'constant', constant_values=0)
        if 'gi50' in X:
            gx = X['gi50'][i][0].reshape(-1)
            ax.set_title(gx)
            #if i==0:
            #    print("x[conc] = {}".format(gx))

        # print('ix = {} {}'.format(ix.shape, ix[0].dtype))
        # print('   = {}'.format(ix[0:5]))
        plt.imshow(ix.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        if 'gi50' in Z:
            gx = Z['gi50'][i][0].reshape(-1)
            ax.set_title(gx)
            #cx = Z['gi50'][i]
            #ax.set_title(cx.reshape(-1))
            #if i==0:
            #    print("c[conc] = {}".format(cx))

        lx = np.array(Z['latent'][i])
        # print('lx = {} {}'.format(lx.shape, lx[0].dtype))
        # print('   = {}'.format(lx[0:5]))
        plt.stem(lx.reshape(-1))
        #plt.text(-0.5, 0, conc_v[i], ha='center', va='center', size=8)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)

        ax = plt.subplot(3, n, i + 1 + n + n)
        oy = np.pad(Y['fp'][i], [(0, 0), (0, 3)], 'constant', constant_values=0)
        # print('oy = {} {}'.format(oy.shape, oy[0].dtype))
        # print('   = {}'.format(oy[0:5]))
        plt.imshow(oy.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save_flag:
        plt.savefig(save_dir + '/eval_e%d.png'%epoch_tag)
    else:
        plt.show()

def show_image_sup(Z, Y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    plt.figure(figsize=(20, 8))
    for i in range(n):
        ax = plt.subplot(5, n / 5, i +1)
        ix = np.pad(Y['fp'][i], ((0, 0), (0, 3)), 'constant', constant_values=0)
        if 'gi50' in Z:
            gx = Z['gi50'][i][0].reshape(-1)
            ax.set_title(gx)
            #if i==0:
            #    print("x[conc] = {}".format(gx))

        # print('ix = {} {}'.format(ix.shape, ix[0].dtype))
        # print('   = {}'.format(ix[0:5]))
        plt.imshow(ix.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        """
        ax = plt.subplot(3, n, i + 1 + n)
        if 'gi50' in Z:
            gx = Z['gi50'][i][0].reshape(-1)
            ax.set_title(gx)
            #cx = Z['gi50'][i]
            #ax.set_title(cx.reshape(-1))
            #if i==0:
            #    print("c[conc] = {}".format(cx))

        lx = np.array(Z['latent'][i])
        # print('lx = {} {}'.format(lx.shape, lx[0].dtype))
        # print('   = {}'.format(lx[0:5]))
        plt.stem(lx.reshape(-1))
        #plt.text(-0.5, 0, conc_v[i], ha='center', va='center', size=8)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)

        ax = plt.subplot(3, n, i + 1 + n + n)
        oy = np.pad(Y['fp'][i], [(0, 0), (0, 3)], 'constant', constant_values=0)
        # print('oy = {} {}'.format(oy.shape, oy[0].dtype))
        # print('   = {}'.format(oy[0:5]))
        plt.imshow(oy.reshape(13, 13))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        """
    if save_flag:
        plt.savefig(save_dir + '/eval_sup_e%d.png'%epoch_tag)
    else:
        plt.show()

def show_image_sum(input_x, latent_z, conc_v, output_y, n=10, save_flag=True, epoch_tag=None, save_dir=None):
    X = np.array(input_x)
    #print("X = {}".format(X.shape))
    X = np.reshape(X, (10, 166))
    #print("X = {}".format(X.shape))
    Y = np.array(output_y)
    #print("Y = {}".format(Y.shape))
    Y = np.reshape(Y, (10, 166))
    #print("Y = {}".format(Y.shape))

    #print("X = {}".format(X))
    #print("Y = {}".format(Y))
    plt.matshow(X-Y)
    plt.savefig(save_dir + '/error_%s.png'%epoch_tag)

    plt.matshow(X)
    plt.savefig(save_dir + '/X_%s.png'%epoch_tag)

def create_colormap(cmap_name='custom', n_bin=255):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    return cm
