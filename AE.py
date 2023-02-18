import tensorflow as tf
import numpy as np
#import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc
from scipy.io import loadmat
from scipy.io import savemat
import math
#import cv2
import sys
tf.reset_default_graph()#后加的！！！
# 输入高光谱数据
hyp_img = loadmat('E:\\AAE\\1\\data\\abu-urban-2-gai_v1_data.mat')#输入deta=0.01，抑制背景
data = hyp_img['X_input']
d = hyp_img['d_1'] #目标向量光谱
[num, bands]=data.shape
print(num,bands)
d_num=1
#data = np.reshape(data,[row*col,bands])
num_examples = num
#输入维度数为波段数
input_dim = bands
n_l1 = 500
n_l2 = 500
z_dim = 15
#每次训练在训练集中取batchsize个样本训练
#1个iteration等于使用batchsize个样本训练一次
#训练集有1000个样本，batchsize=10，那么：训练完整个样本集需要：100次iteration，1次epoch。
batch_size = num
#test_batch_size = testnum
n_epochs = 10000
#learning_rate = 1e-3
learning_rate = 1e-4
#learning_rate_discriminator = 1e-5
beta1 = 0.8
#保存路径
results_path = 'E:\\AAE\\1\\result'
#results_dir = 'E:\\AAE\\1\\testresult'

#if not os.path.exists(results_dir):
#    os.mkdir(results_dir)

x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')
x_vector = tf.placeholder(dtype=tf.float32, shape=[1, input_dim], name='xvector')
decoder_output_vector = tf.placeholder(dtype=tf.float32, shape=[1, input_dim], name='decodervector')

#建立不同结果对应的文件夹
def form_results():
    saved_model_path = results_path  + '/Saved_models/'
    log_path = results_path  + '/log'
    if not os.path.exists(results_path ):
        os.mkdir(results_path )
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return saved_model_path, log_path

#激活函数
def LeakyRelu(x,leaky=0.2,name = 'LeakyRelu'):
    with tf.variable_scope(name, reuse=None):
        f1 = 0.5 * (1 + leaky)
        f2 = 0.5 * (1 - leaky)
        return f1 * x + f2 * tf.abs(x)


#全连接函数
def dense(x, n1, n2, name):
    
    with tf.variable_scope(name, reuse=None):
        #weights 初始化为正态分布
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01)) #分别用于指定均值、标准差、随机数种子和随机数的数据类型
        #bias初始化为常量
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))#偏置项b初始化为0
        #输出为out = x*weight + bias 
        out = tf.add(tf.matmul(x, weights), bias, name='matmul') 
        return out


# 自动编码器网络
# 编码器
# 输出没有使用任何函数激活
def encoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        out1 = dense(x, input_dim, n_l1, 'e_dense_1')
        out2 = dense(out1, n_l1, n_l2, 'e_dense_2')
        out3 = dense(out2, n_l2, z_dim, 'e_latent_variable')
        latent_variable = out3#n_l2
        return latent_variable


#解码器
#输出使用了sigmoid函数激活，以保证输出范围在0--1之间
#输出使用了tanh函数激活
def decoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        out1 = dense(x, z_dim, n_l2, 'd_dense_1')
        out2 = dense(out1, n_l2, n_l1, 'd_dense_2')
        out3 = dense(out2, n_l1, input_dim, 'd_output')
        output = tf.nn.sigmoid(out3)#n_l
        return output


#训练
def train(train_model=True):
    #defen = tf.Variable([0],name = 'defen',dtype=tf.float32)
    
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        decoder_output = decoder(encoder_output)
        
    #二次代价函数
    autoencoder_loss = tf.reduce_mean(tf.square(x_input - decoder_output))
    #autoencoder_Loss(x_target, decoder_output)

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    init = tf.global_variables_initializer()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        if train_model:
            saved_model_path, log_path= form_results() #创建保存结果的文件夹
            sess.run(init)
            
            for i in range(n_epochs+1):
                #这个地方要读取高光谱数据
                n_batches = (int)(num_examples / batch_size)
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                for b in range(n_batches):
                    z_real_dist = np.random.randn(batch_size, z_dim) * 5 #标准正态分布
                    print(b)
                    batch_xr = data
                    batch_x = np.reshape(batch_xr[(b*batch_size):(b*batch_size+batch_size),:],[batch_size,bands])
                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    e_output = sess.run(encoder_output, feed_dict={x_input: batch_x})#输出Z层的值
                    d_output = sess.run(decoder_output,feed_dict={x_input: batch_x})
                    if b % 1 == 0:
                        a_loss= sess.run(
                            [autoencoder_loss],
                            feed_dict={x_input: batch_x, x_target: batch_x,
                                    real_distribution: z_real_dist})
                    
                        #writer.add_summary(summary, global_step=step)
                    print("Epoch: {}, iteration: {}".format(i, b))
                    print("Autoencoder Loss: {}".format(a_loss))

                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(i, b))

                    np.set_printoptions(threshold=np.inf)
                    
                    if i % 10 == 0:
                        decoder_path = 'E:\\AAE\\1\\decoder/'
                        encoder_path = 'E:\\AAE\\1\\encoder/'
                        #d_output_x = tf.transpose(d_output,perm = [1,0])
                        savemat(encoder_path + 'x_encoder%d.mat'%(i), {'x_encoder': e_output})
                        savemat(decoder_path + 'x_decoder%d.mat'%(i), {'x_decoder': d_output})

                step += 1
                saver.save(sess, save_path=saved_model_path, global_step=1)
	    

if __name__ == '__main__':
    train(train_model=True)
    #train(train_model=False) True