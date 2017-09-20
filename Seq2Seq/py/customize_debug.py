from tensorflow.python.debug.lib.debug_data import InconvertibleTensorProto
import numpy as np

def has_nan(datum, tensor):
  """A predicate for whether a tensor consists of any bad numerical values.
  This predicate is common enough to merit definition in this module.
  Bad numerical values include `nan`s and `inf`s.
  The signature of this function follows the requirement of the method
  `DebugDumpDir.find()`.
  Args:
    datum: (`DebugTensorDatum`) Datum metadata.
    tensor: (`numpy.ndarray` or None) Value of the tensor. None represents
      an uninitialized tensor.
  Returns:
    (`bool`) True if and only if tensor consists of any nan or inf values.
  """

  _ = datum  # Datum metadata is unused in this predicate.

  if isinstance(tensor, InconvertibleTensorProto):
    # Uninitialized tensor doesn't have bad numerical values.
    # Also return False for data types that cannot be represented as numpy
    # arrays.
    return False
  elif (np.issubdtype(tensor.dtype, np.float) or
        np.issubdtype(tensor.dtype, np.complex) or
        np.issubdtype(tensor.dtype, np.integer)):
    return np.any(np.isnan(tensor))
  else:
    return False
