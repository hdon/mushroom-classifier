#!/usr/bin/env python
import code, csv, random, time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import pandas as pd
import donpack as dp
from collections_extended import setlist

tf.set_random_seed(time.time())

col_names = setlist((
  'edibility'
, 'cap-shape'
, 'cap-surface'
, 'cap-color'
, 'bruises?'
, 'odor'
, 'gill-attachment'
, 'gill-spacing'
, 'gill-size'
, 'gill-color'
, 'stalk-shape'
, 'stalk-root'
, 'stalk-surface-above-ring'
, 'stalk-surface-below-ring'
, 'stalk-color-above-ring'
, 'stalk-color-below-ring'
, 'veil-type'
, 'veil-color'
, 'ring-number'
, 'ring-type'
, 'spore-print-color'
, 'population'
, 'habitat'
))

raw_data = pd.read_csv(
  'data/agaricus-lepiota.data.txt'
, header=None
, names=col_names
)


raw_x_col_names = col_names - setlist(('edibility',))
raw_y_col_names = col_names - raw_x_col_names

raw_x_values = raw_data.ix[:,raw_x_col_names]
raw_y_values = raw_data.ix[:,raw_y_col_names]

# initialize info about each column as part of preparing our data (x)
x_classes = [(col_name, x) for col_name in raw_x_values.axes[1] for x in raw_data.ix[:,col_name].unique()]
x_class_map = dict()
#x_class_map = dict(((b,a) for a,b in enumerate(x_classes)))

x_values = np.zeros((raw_data.shape[0], len(x_classes)))
for x_index, (raw_col_name, raw_x) in enumerate(x_classes):
  x_class = (raw_col_name, raw_x)
  x_class_map[x_class] = x_index
  x_values[:,x_index] = raw_data.ix[:,raw_col_name] == raw_x

# initialize info about each column as part of preparing our data (y)
y_classes = [(col_name, y) for col_name in raw_y_values.axes[1] for y in raw_data.ix[:,col_name].unique()]
y_class_map = dict()
#y_class_map = dict(((b,a) for a,b in enumerate(y_classes)))

y_values = np.zeros((raw_data.shape[0], len(y_classes)))
for y_index, (raw_col_name, raw_y) in enumerate(y_classes):
  y_class = (raw_col_name, raw_y)
  y_class_map[y_class] = y_index
  y_values[:,y_index] = raw_data.ix[:,raw_col_name] == raw_y

# Hyperparameters
learning_rate = 0.01
training_epochs = 200000
display_step = 10
n_samples = x_values.shape[0]
keep_training = True
epochs = 0
target_cost = 1 # at a cost of 0.0035545605 i had 100% prediction accuracy

# Neural network -- only one layer
layer0_x = tf.placeholder(tf.float32, [None, x_values.shape[1]], name="x0")
layer0_W = tf.Variable(tf.random_normal([x_values.shape[1], y_values.shape[1]], stddev=1), name="W0")
layer0_b = tf.Variable(tf.zeros([y_values.shape[1]]), name="b0")
layer0_y = tf.nn.softmax(tf.add(tf.matmul(layer0_x, layer0_W), layer0_b), name="y0")

y = layer0_y
y_ = tf.placeholder(tf.float32, [None, y_values.shape[1]], name="y_")
cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * x_values.shape[1])
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# summaries for tensorboard
tf.summary.scalar('cost', cost)
summary_op = tf.summary.merge_all()

# intra_op_parallelism_threads can limit our CPU use
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
  init = tf.initialize_all_variables()
  sess.run(init)

  writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())

  while keep_training:
    try:
      sess.run(optimizer, feed_dict={layer0_x: x_values, y_: y_values})
      if epochs % display_step == 0:
        summary, training_cost = sess.run([summary_op, cost], feed_dict={layer0_x: x_values, y_: y_values})
        print 'Training step: % 7d / % 7d' % (epochs, training_epochs), 'cost={:.9f}'.format(training_cost)
        writer.add_summary(summary, epochs)

        if training_cost <= target_cost:
          print 'target cost achieved, no more training'
          break
      epochs += 1
    except KeyboardInterrupt as e:
      print 'error error error'
      print e
      code.interact(local=locals())

  print 'Optimization finished!'
  final_cost = sess.run(cost, feed_dict={layer0_x: x_values, y_: y_values})
  print 'Final cost=', training_cost #, 'W=', sess.run(W), 'b=', sess.run(b)

  #code.interact(local=locals())
