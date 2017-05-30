import pickle, csv, numpy as np, tensorflow as tf
import pandas as pd
from collections_extended import setlist

def load_csv(src, x_col_names, y_col_names, includeColumnNames=False):
  '''Load a csv, normalize its data to a [-1,+1] range using a linear
  function. If any column contains empty values, a "null flag" counterpart
  column is added named "x_nonnull" where "x" was the original field name.
  
  Two numpy arrays are returned, one containing data from the columns named
  in the argument "x_col_names" and one containing data from the columns
  named "y_col_names."

  The fourth argument "includeColumnNames" will cause load_csv to return
  two additional values, each containing the "final" x and y column names,
  which include synthesized columns such as "x_nonnull" columns.
  '''
  df = pd.read_csv(src)
  #print 'load_csv starting out with col count', len(df.columns)
  for col_name in df.columns:
    if np.any(np.isnan(df.loc[:, col_name])):
      #print 'load_csv found nullable column', col_name
      new_col_name = col_name+'_nonnull'
      df.ix[:, new_col_name] = np.where(np.isnan(df.ix[:, col_name]), 1.0, -1.0)
      df.ix[np.isnan(df.ix[:, col_name]), col_name] = 0.0
      x_col_names.append(new_col_name)
    lo = float(df.ix[:, col_name].min())
    hi = float(df.ix[:, col_name].max())
    df.ix[:, col_name] = (df.ix[:, col_name] - lo) / (hi - lo) * 2 - 1
  #print 'load_csv completing with col count', len(df.columns)
  x_matrix = df.as_matrix(x_col_names)
  y_matrix = df.as_matrix(y_col_names)
  if includeColumnNames:
    return x_matrix, y_matrix, x_col_names, y_col_names
  else:
    return x_matrix, y_matrix

def load_csv2(src, x_col_names, y_col_names, includeColumnNames=False):
  '''Load a csv, normalize its data to a mean of zero and then divided by
  the standard deviation of the column. If any column contains empty
  values, a "null flag" counterpart column is added named "x_nonnull" where
  "x" was the original field name.
  
  Two numpy arrays are returned, one containing data from the columns named
  in the argument "x_col_names" and one containing data from the columns
  named "y_col_names."

  The fourth argument "includeColumnNames" will cause load_csv to return
  two additional values, each containing the "final" x and y column names,
  which include synthesized columns such as "x_nonnull" columns.
  '''
  df = pd.read_csv(src)
  #print 'load_csv starting out with col count', len(df.columns)
  for col_name in df.columns:
    if np.any(np.isnan(df.loc[:, col_name])):
      #print 'load_csv found nullable column', col_name
      new_col_name = col_name+'_nonnull'
      df.ix[:, new_col_name] = np.where(np.isnan(df.ix[:, col_name]), 1.0, -1.0)
      df.ix[np.isnan(df.ix[:, col_name]), col_name] = 0.0
      x_col_names.append(new_col_name)
    df.ix[:, col_name] -= np.mean(df.ix[:, col_name])
    df.ix[:, col_name] /= np.stddev(df.ix[:, col_name])
  #print 'load_csv completing with col count', len(df.columns)
  x_matrix = df.as_matrix(x_col_names)
  y_matrix = df.as_matrix(y_col_names)
  if includeColumnNames:
    return x_matrix, y_matrix, x_col_names, y_col_names
  else:
    return x_matrix, y_matrix

def softmax(x):
  # thanks http://stackoverflow.com/questions/34968722/softmax-function-python
  # FYI just watched this video https://www.youtube.com/watch?v=mlaLLQofmR8
  # and his definition of softmax (at 4m7s) is different
  # his definitionw in tf would probably just be this:
  # return np.exp(x) / np.sum(np.exp(x))
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def saveSomeVars(filename, **Vars):
  with open(filename, 'wb') as f:
    pickle.dump({k: Vars[k].eval() for k in Vars}, f)

def loadSomeVars(filename, sess, **Vars):
  num_vars_loaded = 0
  with open(filename, 'rb') as f:
    data = pickle.load(f)
    for k in Vars:
      if k not in data:
        print 'warning: key %s present in your arguments but not in your file %s' % (k, filename)
        continue
      print 'loading data for variable %s' % k
      sess.run(tf.assign(Vars[k], data[k]))
      num_vars_loaded += 1
  print 'loaded %d vars' % num_vars_loaded

#def df2np1hot(df, cols):
#  raw_x_values = df.ix[:,cols]
#  x_classes = [(col_name, x) for col_name in raw_x_values.axes[1] for x in df.ix[:,col_name].unique()]
#  x_class_map = dict()
#  x_values = np.zeros((df.shape[0], len(x_classes)))
#  for x_index, (raw_col_name, raw_x) in enumerate(x_classes):
#    x_class = (raw_col_name, raw_x)
#    x_class_map[x_class] = x_index
#    x_values[:,x_index] = df.ix[:,raw_col_name] == raw_x

__all__ = ['load_csv', 'softmax', 'saveSomeVars', 'loadSomeVars']
