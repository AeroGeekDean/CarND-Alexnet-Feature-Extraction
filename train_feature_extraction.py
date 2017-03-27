import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.misc import imread
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
  data = pickle.load(f)

X_data, y_data = data['features'], data['labels']
n_classes = len(np.unique(y_data))

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
x_resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, n_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix

fc8w = tf.Variable(tf.truncated_normal(shape), name='fc8w')
fc8b = tf.Variable(tf.zeros(n_classes), name='fc8b')

logits = tf.matmul(fc7, fc8w) + fc8b
probs = tf.nn.softmax(logits)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
cost_function = tf.reduce_mean(cross_entropy)

rate = 0.003
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data, sess):
    n_samples = len(X_data)
    total_accuracy = 0
    # sess = tf.get_default_session()
    for offset in range(0, n_samples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        x_batch, y_batch = X_train[offset:end], y_train[offset:end]
        batch_accuracy = sess.run(accuracy_operation, feed_dict={x: x_batch, y: y_batch})
        total_accuracy += (batch_accuracy*len(x_batch))
    return total_accuracy/n_samples

# TODO: Train and evaluate the feature extraction model.
EPOCHS = 10
BATCH_SIZE = 128

from sklearn.utils import shuffle

saver = tf.train.Saver()
save_file = './saved_models/Saved_model'
init = tf.global_variables_initializer()
n_samples = len(X_train)

sess = tf.Session()

if not os.path.isfile(save_file+'9.ckpt.index'):
  sess.run(init)
  print('Training...')
  print()
  for i in range(EPOCHS):
    t0 = time.time()
    print("EPOCH {} ...".format(i))
    # train model
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, n_samples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        # print('\tBatch {}-{} ...'.format(offset, end))
        x_batch, y_batch = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: x_batch, y: y_batch})

    print('time elapsed: {}'.format(time.time()-t0))
    print('Saving model')
    saver.save(sess,save_file+str(i)+'.ckpt')

    print('time elapsed: {}'.format(time.time()-t0))
    print('Calculating accuracy')
    training_accuracy = evaluate(X_train, y_train, sess)
    testing_accuracy = evaluate(X_test, y_test, sess)
    print()
    print("Training Accuracy = {:.3f}, Validation Accuracy = {:.3f}".format(training_accuracy, testing_accuracy))

    # saver.save(sess, './saved_models/lenet'+str(i+1))
    # print("Model-"+str(i+1)+" saved")
    print('time elapsed: {}'.format(time.time()-t0))
    print()
else:
  print('Restoring saved model...')
  saver.restore(sess, './'+save_file+'9.ckpt')

# Now let's evaluate based on the previous images...
# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
