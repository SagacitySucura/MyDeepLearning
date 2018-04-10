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
        new_data.append(new_line[4:])
    return np.array(new_label),np.array(new_data)

#截断正态分布生成随机权重和偏置
def weight_variable(shape,name1):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial,name=name1)
def bias_variable(shape,name1):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name1)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 588])
keep_prob=tf.placeholder(tf.float32)

x=tf.nn.dropout(x,keep_prob)
W01 = weight_variable([588, 100],'W01')
b01 = bias_variable([100],'b01')
hidden1=tf.nn.relu(tf.matmul(x, W01) + b01)


W02 = weight_variable([100, 100],'W02')
b02 = bias_variable([100],'b02')
hidden2=tf.nn.relu(tf.matmul(hidden1, W02) + b02)


W03 = weight_variable([100, 100],'W03')
b03 = bias_variable([100],'b03')
hidden3=tf.nn.relu(tf.matmul(hidden2, W03) + b03)

hidden3=hidden3+hidden1
W04 = weight_variable([100, 100],'W04')
b04 = bias_variable([100],'b04')
hidden4=tf.nn.relu(tf.matmul(hidden3, W04) + b04)

W05 = weight_variable([100, 100],'W05')
b05 = bias_variable([100],'b05')
hidden5=tf.nn.relu(tf.matmul(hidden4, W05) + b05)

W06 = weight_variable([100, 100],'W06')
b06 = bias_variable([100],'b06')
hidden6=tf.nn.relu(tf.matmul(hidden5, W06) + b06)


hidden6=hidden6+hidden3
hidden6=tf.nn.dropout(hidden6,keep_prob)
W07 = weight_variable([100, 100],'W07')
b07 = bias_variable([100],'b07')
hidden7=tf.nn.relu(tf.matmul(hidden6, W07) + b07)

W08 = weight_variable([100, 3],'W08')
b08 = bias_variable([3],'b08')
y = tf.nn.softmax(tf.matmul(hidden7, W08) + b08)


y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)),
                                reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())

for i in range(3300):
    label,data=readData('trainColloction/bondary/train'+str(i)+'.txt',2,588)
    train_step.run({x:data,y_:label,keep_prob:0.85})
probability=0.0
for i in range(800):
    labelT,dataT=readData('testColloction/bondary/test'+str(i)+'.txt',3,588)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    probability=probability+accuracy.eval({x:dataT,y_:labelT[:,0:-1],keep_prob:1.0})
    yout=sess.run(y,feed_dict={x:dataT,keep_prob:1.0})
    new_line=[]
    filehandle=open('result.txt','a',encoding='utf-8')
    for i in range(len(yout)):
        new_line.append(str(yout[i,0]))
        new_line.append(str(yout[i,1]))
        new_line.append(str(labelT[i,2]))
        filehandle.write(','.join(new_line))
        filehandle.write('\n')
        new_line=[]
    filehandle.close()
print(probability/800)



'''
W10 = sess.run(W01)
b10 = sess.run(b01)
W20 = sess.run(W02)
b20 = sess.run(b02)
W30 = sess.run(W03)
b30 = sess.run(b03)
W40 = sess.run(W04)
b40 = sess.run(b04)
np.save('w10',W10)
np.save('b10',b10)
np.save('w20',W20)
np.save('b20',b20)
np.save('w30',W30)
np.save('b30',b30)
np.save('w40',W40)
np.save('b40',b40)
'''