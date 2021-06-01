import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class NumpyAggrFactory(factory.WeightedAggregationFactory):
  def __init__(self, numpy_fn):
    self._numpy_fn = numpy_fn

  def create(
          self, value_type: factory.ValueType,
          weight_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    _check_value_type(value_type)
    py_typecheck.check_type(weight_type, factory.ValueType.__args__)

    @computations.tf_computation
    def numpy_bridge(values, weights):
      v = ds_to_array(values)
      w = ds_to_array(weights)
      return tf.reshape(tf.numpy_function(self._numpy_fn, [v, w], tf.float32), tf.shape(v)[1:])

    @computations.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @computations.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(value_type, tff.CLIENTS),
      computation_types.FederatedType(weight_type, tff.CLIENTS))
    def next_fn(state, value, weight):
      weights = tff.federated_collect(weight)

      weighted_mean_value = tff.federated_zip([
        tff.federated_map(numpy_bridge,
                          (tff.federated_collect(tff.federated_map(tff.tf_computation(lambda _: _[idx]), value)),
                           weights))
        for idx in range(len(value_type))
      ])
      no_metrics = tff.federated_value((), tff.SERVER)
      return measured_process.MeasuredProcessOutput(
        state, weighted_mean_value, no_metrics)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _check_value_type(value_type):
  py_typecheck.check_type(value_type, factory.ValueType.__args__)
  if not type_analysis.is_structure_of_floats(value_type):
    raise TypeError(f'All values in provided value_type must be of floating '
                    f'dtype. Provided value_type: {value_type}')


@tf.function
def ds_to_array(dataset):
  # annoyingly tf graph mode requires the following
  # in order to extract a tensor from the dataset
  # + it requires int64 converting to int32 in the TensorArray API
  tensor_array = tf.TensorArray(dtype=dataset.element_spec.dtype,
                                size=tf.cast(tf.data.experimental.cardinality(dataset), tf.int32),
                                element_shape=dataset.element_spec.shape)
  for i, t in dataset.enumerate():
    tensor_array = tensor_array.write(tf.cast(i, tf.int32), t)
  return tensor_array.stack()
