# prepairing cipher-10 images for machine-learnings
path1 = "where you saved /Cipher-10/cifar-10-batches-py/"

def unpickle(file):
    fp = open(file, 'rb')
    data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data
    
file1 = path1 + "data_batch_1"
file2 = path1 + "data_batch_2"
file3 = path1 + "data_batch_3"
file4 = path1 + "data_batch_4"
file5 = path1 + "data_batch_5"

dataCifar01 = unpickle(file1)
dataCifar02 = unpickle(file2)
dataCifar03 = unpickle(file3)
dataCifar04 = unpickle(file4)
dataCifar05 = unpickle(file5)

dataCifardata = np.concatenate((dataCifar01['data'],dataCifar02['data'],
                                dataCifar03['data'],dataCifar04['data'],dataCifar05['data']),axis=0)
dataCifarlabels = np.concatenate((dataCifar01['labels'],dataCifar02['labels'],
                                dataCifar03['labels'],dataCifar04['labels'],dataCifar05['labels']),axis=0)
                                
                                
datadata = dataCifar10['data']

def next_batcher(data_data,data_labels):
    data_labels = np.array(data_labels)
    x = np.random.permutation(50000)
    data_labels_oneHot_pre = data_labels[x][:100]
    data_labels_one_hot = np.eye(10)[data_labels_oneHot_pre]
    return data_data[x][:100], data_labels_one_hot
    
def image_transposer(data_data):
    data_data_t_pre = np.array(data_data,dtype=np.float32)
    data_data_t = data_data_t_pre.reshape(data_data_t_pre.shape[0], 3, 32, 32)
    data_data_t = data_data_t.transpose([0,2,3,1])
    return data_data_t
    
test_images = image_transposer(batchingdata)

# constructing NN using tesorflow
num_units1 = 3072
num_units2 = 8192
num_units3 = 2048
num_units4 = 2048
num_units5 = 1024
num_units6 = 10
init_const_num = 0.25
num_filters1 = 288
num_filters2 = 72
num_filters3 = 36
num_filters4 = 12

stddev001 = mt.sqrt(2 / num_filters1)
stddev002 = mt.sqrt(2 / num_filters2)
stddev003 = mt.sqrt(2 / num_filters3)
stddev004 = mt.sqrt(2 / num_filters4)
stddev02 = mt.sqrt(2 / num_units1)
stddev03 = mt.sqrt(2 / num_units2)
stddev04 = mt.sqrt(2 / num_units3)
stddev05 = mt.sqrt(2 / num_units4)
stddev06 = mt.sqrt(2 / num_units5)

# x_image = tf.placeholder(tf.float32, [None, 784])
x_imageInUse = tf.placeholder(tf.float32, [None,32,32,3])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,3,num_filters1],stddev=stddev001))
h_conv1 = tf.nn.conv2d(x_imageInUse, W_conv1, strides=[1,2,2,1], padding='SAME')
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5,5,num_filters1,num_filters2]
                                          ,stddev=stddev002))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

W_conv3 = tf.Variable(tf.truncated_normal([5,5,num_filters2,num_filters3],
                                          stddev=stddev003))
h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='SAME')
h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

W_conv4 = tf.Variable(tf.truncated_normal([5,5,num_filters3,num_filters4],
                                          stddev=stddev004))
h_conv4 = tf.nn.conv2d(h_pool3, W_conv4, strides=[1,1,1,1], padding='SAME')
h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

h_pool_flat = tf.reshape(h_pool4, [-1,num_filters4])

w4 = tf.Variable(tf.truncated_normal([num_filters4, num_units1],stddev=stddev004))
b4 = tf.Variable(tf.constant(init_const_num, shape=[num_units1]))
hidden4 = tf.nn.relu(tf.matmul(h_pool_flat, w4) + b4)

w3 = tf.Variable(tf.truncated_normal([num_units1, num_units2],stddev=stddev02))
b3 = tf.Variable(tf.constant(init_const_num, shape=[num_units2]))
hidden3 = tf.nn.relu(tf.matmul(hidden4, w3) + b3)

w2 = tf.Variable(tf.truncated_normal([num_units2, num_units3],stddev=stddev03))
b2 = tf.Variable(tf.constant(init_const_num, shape=[num_units3]))
hidden2 = tf.nn.relu(tf.matmul(hidden3, w2) + b2)

w1_5 = tf.Variable(tf.truncated_normal([num_units3, num_units4],stddev=stddev04))
b1_5 = tf.Variable(tf.constant(init_const_num, shape=[num_units4]))
hidden1_5 = tf.nn.relu(tf.matmul(hidden2, w1_5) + b1_5)

w1 = tf.Variable(tf.truncated_normal([num_units4, num_units5],stddev=stddev05))
b1 = tf.Variable(tf.constant(init_const_num, shape=[num_units5]))
hidden1 = tf.nn.relu(tf.matmul(hidden1_5, w1) + b1)

w0 = tf.Variable(tf.truncated_normal([num_units5, num_units6],stddev=stddev06))
b0 = tf.Variable(tf.constant(init_const_num, shape=[num_units6]))
p = tf.nn.relu(tf.matmul(hidden1, w0) + b0)

# constructing optimizers and others
t = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=p))
train_step = tf.train.AdamOptimizer(0.0003).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# trying to use tensor board
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("weights_hidden",w1)
tf.summary.histogram("biases_hidden",b1)
#tf.summary.histogram("weights_hidden2",w2)
#tf.summary.histogram("biases_hidden2",b2)
#tf.summary.histogram("weights_hidden3",w3)
#tf.summary.histogram("biases_hidden3",b3)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("tf_log2017-08-31", graph=sess.graph)

# initializeing sess
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# training
i = 0
for _ in range(2000):
    i += 1
    xs, ts = next_batcher(dataCifardata,dataCifarlabels)
    xs = image_transposer(xs)
    sess.run(train_step,feed_dict={x_imageInUse:xs, t:ts})
    if i % 50 == 0:
        # xs, tsii = next_batcher(dataCifar10)
        # x = image_transposer(xs)
        #loss_val, acc_val, summary = sess.run([loss, accuracy, summary_op], feed_dict={x_imageInUse:xs, t:ts})
        train_accuracy = accuracy.eval(feed_dict={x_imageInUse:xs, t:ts})
        print("step %d, training accuracy %f" % (i, train_accuracy))
        #summary_writer.add_summary(summary,i)
        #print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        saver.save(sess, 'cifar2017-08-31', global_step=i)
