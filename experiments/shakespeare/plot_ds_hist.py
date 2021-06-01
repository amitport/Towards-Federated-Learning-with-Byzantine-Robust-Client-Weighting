import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from utils.datasets.shakespeare_dataset import _build_tokenize_fn, SEQUENCE_LENGTH

from pathlib import Path

expr_out_path = Path.home() / 'expr_out'


expr_out_path.mkdir(parents=True, exist_ok=True)

def hist(parts, title, bins=100):
  k = len(parts)
  fig = plt.figure(figsize=(5, 3))

  plt.rc('axes', labelsize=10)  # fontsize of the x and y labels

  ax = plt.gca()
  ax.set_xlabel('Client sample size')
  ax.set_ylabel('Frequency')

  ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))  # FuncFormatter('{0:.0%}'.format))

  plt.title('Shakespeare')

  plt.hist(parts, bins=bins, weights=np.repeat(1 / k, k))
  plt.tight_layout()

  fig.savefig(expr_out_path / f'{title}_partition.pdf', format='pdf')
  plt.show()


def pint_hist(fed_ds, counter, title):
  sizes = []
  for client_id in fed_ds.client_ids:
    ds = fed_ds.create_tf_dataset_for_client(client_id)
    sizes.append(ds.reduce(tf.constant(0, tf.int64), counter).numpy())

  # first values in shakespeare should be:
  # [2450.0, 462.0, 1999.0, 1076.0, 891.0, 50.0, 663.0, 48.0, 31655.0, 4497.0]

  hist(sizes, title)


# datasets from LEAF https://arxiv.org/abs/1812.01097

# shakespeare - next-character prediction
# =======================================
# The data set consists of 715 users (characters of Shakespeare plays),
# where each example corresponds to a contiguous set of lines spoken
# by the character in a given play.
# train: 16,068 examples
# test: 2,356 examples
# training from McMahan et al AISTATS 2017
shakespeare_train, shakespeare_test = tff.simulation.datasets.shakespeare.load_data()

to_tokens = _build_tokenize_fn()


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64), shakespeare_train.element_type_structure])
def shakespeare_tokens_counter(accum, _):
  # match the setup from training from McMahan et al AISTATS 2017
  tokens = to_tokens(_)

  # ignore padding (zeros) and account for shift per sequence (SEQUENCE_LENGTH)
  return accum + tf.math.count_nonzero(tokens) - tf.cast(tf.size(tokens) // SEQUENCE_LENGTH, tf.int64)


pint_hist(shakespeare_train, shakespeare_tokens_counter, 'Shakespeare_tokens')
