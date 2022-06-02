import unittest
import autodidax as dax
import numpy as np
import jax
import jax.numpy as jnp

dax.jax_types.add(jnp.DeviceArray)
# dax.jax_types.add(jax.interpreters.ad.JVPTracer)

def f(lib, x):
  y = lib.sin(x) * 2.
  z = - y + x
  return z

def g(x):
  y = reduce_sum(x, axis=0)
  return y

class AutodidaxTestCase(dax.test_util.DaxTestCase):
  def test_basic(self):
    a = 3.0
    b = np.ones((3, 3))
    self.assertAllClose(f(dax, a), f(np, a))
    self.assertAllClose(f(dax, b), f(np, b))
  def test_jvp_v1(self):
    # And with that, we can differentiate!
    jvp_v1, sin, cos = dax.jvp_v1, dax.sin, dax.cos

    x = 3.0
    y, sin_deriv_at_3 = jvp_v1(sin, (x,), (1.0,))
    y_, sin_deriv_at_3_ = jax.jvp(jnp.sin, (x,), (1.0,))
    self.assertAllClose(sin_deriv_at_3, cos(x))
    self.assertAllClose(sin_deriv_at_3, sin_deriv_at_3_)

    # +
    def f(x):
      y = sin(x) * 2.
      z = - y + x
      return z
    def f_(x):
      y = jnp.sin(x) * 2.
      z = - y + x
      return z

    x, xdot = 3., 1.
    y, ydot = jvp_v1(f, (x,), (xdot,))
    y_, ydot_ = jax.jvp(f_, (x,), (xdot,))
    # print(y)
    # print(ydot)


    # +
    def deriv(f):
      return lambda x: jvp_v1(f, (x,), (1.,))[1]
    def deriv_(f):
      return lambda x: jax.jvp(f, (x,), (1.,))[1]

    # print(deriv(sin)(3.))
    # print(deriv(deriv(sin))(3.))
    # print(deriv(deriv(deriv(sin)))(3.))
    # print(deriv(deriv(deriv(deriv(sin))))(3.))
    self.assertAllClose(deriv(sin)(3.), deriv_(jnp.sin)(3.))
    self.assertAllClose(deriv(deriv(sin))(3.), deriv_(deriv_(jnp.sin))(3.))
    self.assertAllClose(deriv(deriv(deriv(sin)))(3.), deriv_(deriv_(deriv_(jnp.sin)))(3.))


    # +
    def f(x):
      if x > 0.:  # Python control flow
        return 2. * x
      else:
        return x

    # print(deriv(f)(3.))
    # print(deriv(f)(-3.))
    self.assertAllClose(deriv(f)(3.), deriv_(f)(3.))
    self.assertAllClose(deriv(f)(-3.), deriv_(f)(-3.))


