import numpy as np
import unittest

from autodidax._src import dtypes as _dtypes
from autodidax._src.config import flags, bool_env, config

EPS = 1e-4

def _dtype(x):
  return (getattr(x, 'dtype', None) or
          np.dtype(_dtypes.python_scalar_dtypes.get(type(x), None)) or
          np.asarray(x).dtype)


def num_float_bits(dtype):
  return _dtypes.finfo(_dtypes.canonicalize_dtype(dtype)).bits


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True

_default_tolerance = {
  _dtypes.float0: 0,
  np.dtype(np.bool_): 0,
  np.dtype(np.int8): 0,
  np.dtype(np.int16): 0,
  np.dtype(np.int32): 0,
  np.dtype(np.int64): 0,
  np.dtype(np.uint8): 0,
  np.dtype(np.uint16): 0,
  np.dtype(np.uint32): 0,
  np.dtype(np.uint64): 0,
  np.dtype(_dtypes.bfloat16): 1e-2,
  np.dtype(np.float16): 1e-3,
  np.dtype(np.float32): 1e-6,
  np.dtype(np.float64): 1e-15,
  np.dtype(np.complex64): 1e-6,
  np.dtype(np.complex128): 1e-15,
}

def default_tolerance():
  if device_under_test() != "tpu":
    return _default_tolerance
  tol = _default_tolerance.copy()
  tol[np.dtype(np.float32)] = 1e-3
  tol[np.dtype(np.complex64)] = 1e-3
  return tol

default_gradient_tolerance = {
  np.dtype(_dtypes.bfloat16): 1e-1,
  np.dtype(np.float16): 1e-2,
  np.dtype(np.float32): 2e-3,
  np.dtype(np.float64): 1e-5,
  np.dtype(np.complex64): 1e-3,
  np.dtype(np.complex128): 1e-5,
}

def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  if a.dtype == b.dtype == _dtypes.float0:
    np.testing.assert_array_equal(a, b, err_msg=err_msg)
    return
  a = a.astype(np.float32) if a.dtype == _dtypes.bfloat16 else a
  b = b.astype(np.float32) if b.dtype == _dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  with np.errstate(invalid='ignore'):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)

def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {np.dtype(key): value for key, value in tol.items()}
  dtype = _dtypes.canonicalize_dtype(np.dtype(dtype))
  return tol.get(dtype, default_tolerance()[dtype])

def _normalize_tolerance(tol):
  tol = tol or 0
  if isinstance(tol, dict):
    return {np.dtype(k): v for k, v in tol.items()}
  else:
    return {k: tol for k in _default_tolerance}

def join_tolerance(tol1, tol2):
  tol1 = _normalize_tolerance(tol1)
  tol2 = _normalize_tolerance(tol2)
  out = tol1
  for k, v in tol2.items():
    out[k] = max(v, tol1.get(k, 0))
  return out

def _assert_numpy_close(a, b, atol=None, rtol=None, err_msg=''):
  a, b = np.asarray(a), np.asarray(b)
  assert a.shape == b.shape
  atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
  rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
  _assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size,
                         err_msg=err_msg)


class DaxTestCase(unittest.TestCase):
  def assertArraysEqual(self, x, y, *, check_dtypes=True, err_msg=''):
    """Assert that x and y arrays are exactly equal."""
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    # Work around https://github.com/numpy/numpy/issues/18992
    with np.errstate(over='ignore'):
      np.testing.assert_array_equal(x, y, err_msg=err_msg)

  def assertArraysAllClose(self, x, y, *, check_dtypes=True, atol=None,
                           rtol=None, err_msg=''):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    if not config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(_dtypes.canonicalize_dtype(_dtype(x)),
                       _dtypes.canonicalize_dtype(_dtype(y)))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, *, check_dtypes=True, atol=None, rtol=None,
                     canonicalize_dtypes=True, err_msg=''):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif hasattr(x, '__array__') or np.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or np.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = np.asarray(x)
      y = np.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
                                err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))


def device_under_test():
  return "cpu"