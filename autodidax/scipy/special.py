from autodidax import lax

def logsumexp(x, axis=None, keepdims=False):
    c = lax.max(x, axis=axis, keepdims=keepdims)
    return c + lax.log(lax.sum(lax.exp(x - c), axis=axis, keepdims=keepdims))

# @_wraps(osp_special.logsumexp)
# def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
#   if b is not None:
#     a, b = _promote_args_inexact("logsumexp", a, b)
#     a = jnp.where(b != 0, a, -jnp.inf)
#   else:
#     a, = _promote_args_inexact("logsumexp", a)
#   pos_dims, dims = _reduction_dims(a, axis)
#   amax = jnp.max(a, axis=dims, keepdims=keepdims)
#   amax = lax.stop_gradient(lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0)))
#   amax_with_dims = amax if keepdims else lax.expand_dims(amax, pos_dims)
#   # fast path if the result cannot be negative.
#   if b is None and not np.issubdtype(a.dtype, np.complexfloating):
#     out = lax.add(lax.log(jnp.sum(lax.exp(lax.sub(a, amax_with_dims)),
#                                   axis=dims, keepdims=keepdims)),
#                   amax)
#     sign = jnp.where(jnp.isnan(out), out, 1.0)
#     sign = jnp.where(jnp.isneginf(out), 0.0, sign).astype(out.dtype)
#   else:
#     expsub = lax.exp(lax.sub(a, amax_with_dims))
#     if b is not None:
#       expsub = lax.mul(expsub, b)
#     sumexp = jnp.sum(expsub, axis=dims, keepdims=keepdims)
#
#     sign = lax.stop_gradient(jnp.sign(sumexp))
#     if np.issubdtype(sumexp.dtype, np.complexfloating):
#       if return_sign:
#         sumexp = sign*sumexp
#       out = lax.add(lax.log(sumexp), amax)
#     else:
#       out = lax.add(lax.log(lax.abs(sumexp)), amax)
#   if return_sign:
#     return (out, sign)
#   if b is not None:
#     if not np.issubdtype(out.dtype, np.complexfloating):
#       with jax.debug_nans(False):
#         out = jnp.where(sign < 0, jnp.array(np.nan, dtype=out.dtype), out)
#   return out

