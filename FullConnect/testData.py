import tensorflow as tf
import numpy as np
import os
import linecache
rootdir="H:/deep_learning/ringAtListOne/deep_learning/allconnectnet/coord_data/"
os.chdir(rootdir)

def readData(filename,labellen,datalen):
    lines=linecache.getlines(filename)
    linecache.clearcache()
    new_label=[]
    new_data=[]
    for line in lines:
        new_line=[float(i) for i in line.split(',')]
        new_label.append(new_line[0:labellen])
        new_data.append(new_line[(labellen+1):])
    return np.array(new_label),np.array(new_data)

keep_prob=tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 588])
W01 = np.reshape(np.load('w10.npy'),[588,200])
b01 = np.reshape(np.load('b10.npy'),[200])
hidden = tf.nn.relu(tf.matmul(x, W01) + b01)
hidden_drop=tf.nn.dropout(hidden,keep_prob)
W02 = np.reshape(np.load('w20.npy'),[200,100])
b02 = np.reshape(np.load('b20.npy'),[100])
hidden2= tf.nn.relu(tf.matmul(hidden_drop, W02) + b02)
W03 = np.reshape(np.load('w30.npy'),[100,50])
b03 = np.reshape(np.load('b30.npy'),[50])
hidden3= tf.nn.relu(tf.matmul(hidden2, W03) + b03)
W04 = np.reshape(np.load('w40.npy'),[50,2])
b04 = np.reshape(np.load('b40.npy'),[2])
y = tf.nn.softmax(tf.matmul(hidden3, W04) + b04)



label,data=readData('trainDataRandom.txt',2,588)
sess=tf.InteractiveSession()
y_=tf.placeholder(tf.float32,[None,2])
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:data,y_:label,keep_prob:1}))





'''
yout=sess.run(y,feed_dict={x:new_x,keep_prob:1})
filehandle=open('reslut.txt','a')
for i in range(len(yout)):
    for j in range(len(yout[i])):
        filehandle.write(str(yout[i,j])+"   ")
    filehandle.write(str(new_real[i]))
    filehandle.write('\n')
filehandle.close()
print(accuracy.eval({x:new_x,y_:new_y,keep_prob:1.0}))'''

