# This file was adapted from https://github.com/google-research/federated:
#
# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import functools
import os.path
from typing import Callable
import numpy as np

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning import ClientWeighting

from shared.aggregators import trimmed_mean, median, mean
from shared.truncate import find_U
from optimization.shared import training_specs
from optimization.shared import optimizer_utils
from experiments.shakespeare import federated_shakespeare, federated_stackoverflow
import experiments.shakespeare.tff_patch as tff_patch
from experiments.shakespeare.numpy_aggr import NumpyAggrFactory
from experiments.shakespeare.robust_aggregation import RobustWeiszfeldFactory, TruncatedRobustWeiszfeldFactory
from experiments.shakespeare.client_weight import get_client_weights
from utils import training_loop
from utils import utils_impl


_SUPPORTED_TASKS = [
  'shakespeare', 'stackoverflow',
]

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_string(
    'experiment_name', None, 'The name of this experiment. Will be append to '
                             '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
    'rounds_per_eval', 1,
    'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

  # Parameters specific for our paper
  flags.DEFINE_enum('weight_preproc', 'passthrough', ['passthrough', 'ignore', 'uniform', 'truncate'],
                    'What to do with the clients\' relative weights.')
  # flags.DEFINE_float('weight_truncate_U', None, 'truncate threshold when weight_preproc is \'truncate\'')

  flags.DEFINE_enum('aggregation', 'mean', ['mean', 'trimmed_mean', 'median', 'rfa'], 'select aggregation type to use')

  flags.DEFINE_enum('attack', 'none', ['none', 'delta_to_zero'], 'select attack type')
  flags.DEFINE_enum('num_byzantine', '10_percent', ['10_percent', 'single'], 'select the number of byzantine clients')
  flags.DEFINE_integer('byzantine_client_weight', 1_000_000, 'fake client weight byzantine client publish')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

with utils_impl.record_hparam_flags() as shakespeare_flags:
  # Shakespeare flags
  flags.DEFINE_integer(
    'shakespeare_sequence_length', 80,
    'Length of character sequences to use for the RNN model.')


with utils_impl.record_hparam_flags() as stackoverflow_flags:
  # Stack Overflow flags
  flags.DEFINE_integer(
    'stackoverflow_vocab_size', 10000,
    'Integer dictating the number of most frequent words to use in the vocabulary.')
  flags.DEFINE_integer(
    'stackoverflow_num_oov_buckets', 1,
    'The number of out-of-vocabulary buckets to use.')
  flags.DEFINE_integer(
    'stackoverflow_sequence_length', 20,
    'The maximum number of words to take for each sequence.')
  flags.DEFINE_integer(
    'stackoverflow_max_elements_per_user', 1000,
    "The maximum number of elements processed for each client's dataset.")
  flags.DEFINE_integer(
    'stackoverflow_num_validation_examples', 10000,
    'The number of test examples to use for validation.')
  flags.DEFINE_integer(
    'stackoverflow_embedding_size', 96,
    'The dimension of the word embedding layer.')
  flags.DEFINE_integer(
    'stackoverflow_latent_size', 670,
    'The dimension of the latent units in the recurrent layers.')
  flags.DEFINE_integer(
    'stackoverflow_num_layers', 1,
    'The number of stacked recurrent layers to use.')
  flags.DEFINE_bool(
    'stackoverflow_shared_embedding', False,
    'Boolean indicating whether to tie input and output embeddings.')

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
  shakespeare=shakespeare_flags,
  stackoverflow=stackoverflow_flags,
)


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  # If GPU is provided, TFF will by default use the first GPU like TF. The
  # following lines will configure TFF to use multi-GPUs and distribute client
  # computation on the GPUs. Note that we put server computatoin on CPU to avoid
  # potential out of memory issue when a large number of clients is sampled per
  # round. The client devices below can be an empty list when no GPU could be
  # detected by TF.
  # client_devices = tf.config.list_logical_devices('GPU')
  # server_device = tf.config.list_logical_devices('CPU')[0]
  # tff.backends.native.set_local_execution_context(
  #     server_tf_device=server_device, client_tf_devices=client_devices)

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  def iterative_process_builder(
          model_fn: Callable[[],
                             tff.learning.Model]) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """
    if FLAGS.task == 'shakespeare' or FLAGS.task == 'stackoverflow':

      def client_weight_fn(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    else:
      client_weight_fn = None

    if FLAGS.weight_preproc == 'ignore':
      def client_weight_fn(local_outputs):
        return tf.constant(1.0, tf.float32)
    elif FLAGS.weight_preproc == 'uniform': # should be the same as ignore (just verifying)
      client_weight_fn = ClientWeighting.UNIFORM

    if FLAGS.aggregation == 'trimmed_mean':
      inner_aggregator = functools.partial(trimmed_mean, beta=0.1)
    elif FLAGS.aggregation == 'median':
      inner_aggregator = median
    else:
      inner_aggregator = mean

    if FLAGS.weight_preproc == 'truncate':
      def aggregate_with_truncation(points, weights):
        U = find_U(weights, alpha_star=0.5, alpha=0.1)
        return inner_aggregator(points, np.where(weights < U, weights, U))

      aggregator = NumpyAggrFactory(aggregate_with_truncation)
    else:
      if FLAGS.aggregation == 'mean':
        aggregator = None  # defaults to reduce mean
      else:
        aggregator = NumpyAggrFactory(inner_aggregator)

    if FLAGS.aggregation == 'rfa':
      if FLAGS.weight_preproc == 'truncate':
        weights = get_client_weights(FLAGS.task)
        weights = list(weights)
        weights = np.array(weights)
        U = find_U(weights, alpha_star=0.5, alpha=0.1)
        aggregator = TruncatedRobustWeiszfeldFactory(U)
      else:
        aggregator = RobustWeiszfeldFactory()

    return tff_patch.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weight_fn,
      # broadcast_process=encoded_broadcast_process,
      model_update_aggregation_factory=aggregator,
      # use_experimental_simulation_loop=True,
      byzantine_client_weight=FLAGS.byzantine_client_weight
    )

  task_spec = training_specs.TaskSpec(
    iterative_process_builder=iterative_process_builder,
    client_epochs_per_round=FLAGS.client_epochs_per_round,
    client_batch_size=FLAGS.client_batch_size,
    clients_per_round=FLAGS.clients_per_round,
    client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  if FLAGS.task == 'shakespeare':
    runner_spec = federated_shakespeare.configure_training(
      task_spec, sequence_length=FLAGS.shakespeare_sequence_length, attack=FLAGS.attack, num_byzantine=FLAGS.num_byzantine)
  elif FLAGS.task == 'stackoverflow':
    runner_spec = federated_stackoverflow.configure_training(
      task_spec,
      vocab_size=FLAGS.stackoverflow_vocab_size, num_oov_buckets=FLAGS.stackoverflow_num_oov_buckets,
      sequence_length=FLAGS.stackoverflow_sequence_length,
      max_elements_per_user=FLAGS.stackoverflow_max_elements_per_user,
      num_validation_examples=FLAGS.stackoverflow_num_validation_examples,
      embedding_size=FLAGS.stackoverflow_embedding_size, latent_size=FLAGS.stackoverflow_latent_size,
      num_layers=FLAGS.stackoverflow_num_layers, shared_embedding=FLAGS.stackoverflow_shared_embedding,
      attack=FLAGS.attack, num_byzantine=FLAGS.num_byzantine)
  else:
    raise ValueError(
      '--task flag {} is not supported, must be one of {}.'.format(
        FLAGS.task, _SUPPORTED_TASKS))

  _write_hparam_flags()

  training_loop.run(
    iterative_process=runner_spec.iterative_process,
    client_datasets_fn=runner_spec.client_datasets_fn,
    validation_fn=runner_spec.validation_fn,
    test_fn=runner_spec.test_fn,
    total_rounds=FLAGS.total_rounds,
    experiment_name=FLAGS.experiment_name,
    root_output_dir=FLAGS.root_output_dir,
    rounds_per_eval=FLAGS.rounds_per_eval,
    rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
