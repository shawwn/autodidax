from typing import NamedTuple

print = lambda *args: None

class Primitive(NamedTuple):
  name: str

add_p = Primitive('add')
mul_p = Primitive('mul')
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")
transpose_p = Primitive("transpose")
broadcast_p = Primitive("broadcast")

def add(x, y): return bind1(add_p, x, y)
def mul(x, y): return bind1(mul_p, x, y)
def neg(x): return bind1(neg_p, x)
def sin(x): return bind1(sin_p, x)
def cos(x): return bind1(cos_p, x)
def reduce_sum(x, axis=None): return bind1(reduce_sum_p, x, axis=axis)
def greater(x, y): return bind1(greater_p, x, y)
def less(x, y): return bind1(less_p, x, y)
def transpose(x, perm): return bind1(transpose_p, x, perm=perm)
def broadcast(x, shape, axes): return bind1(broadcast_p, x, shape=shape, axes=axes)

def bind1(prim, *args, **params):
  out, = bind(prim, *args, **params)
  return out

# ---------------

from contextlib import contextmanager
from typing import Type, List, Tuple, Sequence, Optional, Any

class MainTrace(NamedTuple):
  level: int
  trace_type: Type['Trace']
  global_data: Optional[Any]

trace_stack: List[MainTrace] = []
dynamic_trace: Optional[MainTrace] = None  # to be employed in Part 3

@contextmanager
def new_main(trace_type: Type['Trace'], global_data=None):
  level = len(trace_stack)
  main = MainTrace(level, trace_type, global_data)
  trace_stack.append(main)

  try:
    yield main
  finally:
    trace_stack.pop()

# ---------------

class Trace:
  main: MainTrace

  def __init__(self, main: MainTrace) -> None:
    self.main = main

  def pure(self, val): assert False  # must override
  def lift(self, val): assert False  # must override

  def process_primitive(self, primitive, tracers, params):
    assert False  # must override

# ---------------

class EvalTrace(Trace):
  pure = lift = lambda self, x: x  # no boxing in Tracers needed

  def process_primitive(self, primitive, tracers, params):
    return impl_rules[primitive](*tracers, **params)

trace_stack.append(MainTrace(0, EvalTrace, None))  # special bottom of the stack

# NB: in JAX, instead of a dict we attach impl rules to the Primitive instance
impl_rules = {}

impl_rules[add_p] = lambda x, y: [np.add(x, y)]
impl_rules[mul_p] = lambda x, y: [np.multiply(x, y)]
impl_rules[neg_p] = lambda x: [np.negative(x)]
impl_rules[sin_p] = lambda x: [np.sin(x)]
impl_rules[cos_p] = lambda x: [np.cos(x)]
impl_rules[reduce_sum_p] = lambda x, *, axis: [np.sum(x, axis)]
impl_rules[greater_p] = lambda x, y: [np.greater(x, y)]
impl_rules[less_p] = lambda x, y: [np.less(x, y)]
impl_rules[transpose_p] = lambda x, *, perm: [np.transpose(x, perm)]

def broadcast_impl(x, *, shape, axes):
  for axis in sorted(axes):
    x = np.expand_dims(x, axis)
  return [np.broadcast_to(x, shape)]
impl_rules[broadcast_p] = broadcast_impl

# ---------------

def bind(prim, *args, **params):
  top_trace = find_top_trace(args)
  tracers = [full_raise(top_trace, arg) for arg in args]
  outs = top_trace.process_primitive(prim, tracers, params)
  return [full_lower(out) for out in outs]

# ---------------

import operator as op

def find_top_trace(xs) -> Trace:
  top_main = max((x._trace.main for x in xs if isinstance(x, Tracer)),
                 default=trace_stack[0], key=op.attrgetter('level'))
  if dynamic_trace and dynamic_trace.level > top_main.level:
    top_main = dynamic_trace
  return top_main.trace_type(top_main)

# ---------------

import numpy as np

class Tracer:
  _trace: Trace

  __array_priority__ = 1000

  @property
  def aval(self):
    assert False  # must override

  def full_lower(self):
    return self  # default implementation

  def __neg__(self): return self.aval._neg(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __lt__(self, other): return self.aval._lt(self, other)
  def __bool__(self): return self.aval._bool(self)
  def __nonzero__(self): return self.aval._nonzero(self)

  def __getattr__(self, name):
    try:
      return getattr(self.aval, name)
    except AttributeError:
      raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

def swap(f): return lambda x, y: f(y, x)

# +
class ShapedArray:
  array_abstraction_level = 1
  shape: Tuple[int]
  dtype: np.dtype

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  @property
  def ndim(self):
    return len(self.shape)

  _neg = staticmethod(neg)
  _add = staticmethod(add)
  _radd = staticmethod(swap(add))
  _mul = staticmethod(mul)
  _rmul = staticmethod(swap(mul))
  _gt = staticmethod(greater)
  _lt = staticmethod(less)

  @staticmethod
  def _bool(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  @staticmethod
  def _nonzero(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  def str_short(self):
    return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

  def __hash__(self):
    return hash((self.shape, self.dtype))

  def __eq__(self, other):
    return (type(self) is type(other) and
            self.shape == other.shape and self.dtype == other.dtype)

  def __repr__(self):
    return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"

class ConcreteArray(ShapedArray):
  array_abstraction_level = 2
  val: np.ndarray

  def __init__(self, val):
    self.val = val
    self.shape = val.shape
    self.dtype = val.dtype

  @staticmethod
  def _bool(tracer):
    return bool(tracer.aval.val)

  @staticmethod
  def _nonzero(tracer):
    return bool(tracer.aval.val)

def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  elif type(x) in jax_types:
    return ConcreteArray(np.asarray(x))
  else:
    raise TypeError(x)

jax_types = {bool, int, float,
             np.bool_, np.int32, np.int64, np.float32, np.float64, np.ndarray}

# ---------------

def full_lower(val: Any):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def full_raise(trace: Trace, val: Any) -> Tracer:
  if not isinstance(val, Tracer):
    assert type(val) in jax_types
    return trace.pure(val)
  level = trace.main.level
  if val._trace.main is trace.main:
    return val
  elif val._trace.main.level < level:
    return trace.lift(val)
  elif val._trace.main.level > level:
    raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
  else:  # val._trace.level == level
    raise Exception(f"Different traces at same level: {val._trace}, {trace}.")

# ---------------

def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  elif type(x) in jax_types:
    return ConcreteArray(np.asarray(x))
  else:
    raise TypeError(x)


# ### Forward-mode autodiff with `jvp`
#
# First, a few helper functions:

# +
def zeros_like(val):
  aval = get_aval(val)
  return np.zeros(aval.shape, aval.dtype)

def unzip2(pairs):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  return lst1, lst2

map_ = map
def map(f, *xs):
  return list(map_(f, *xs))

zip_ = zip
def zip(*args):
  fst, *rest = args = map(list, args)
  n = len(fst)
  for arg in rest:
    assert len(arg) == n
  return list(zip_(*args))


# -

# The `Tracer` for forward-mode autodiff carries a primal-tangent pair. The
# `Trace` applies JVP rules.

# +
class JVPTracer(Tracer):
  def __init__(self, trace, primal, tangent):
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    return get_aval(self.primal)

class JVPTrace(Trace):
  pure = lift = lambda self, val: JVPTracer(self, val, zeros_like(val))

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    jvp_rule = jvp_rules[primitive]
    primal_outs, tangent_outs = jvp_rule(primals_in, tangents_in, **params)
    return [JVPTracer(self, x, t) for x, t in zip(primal_outs, tangent_outs)]

jvp_rules = {}


# -

# Notice both `pure` and `lift` package a value into a `JVPTracer` with the
# minimal amount of context, which is a zero tangent value.
#
# Let's add some JVP rules for primitives:

# +
def add_jvp(primals, tangents):
  (x, y), (x_dot, y_dot) = primals, tangents
  return [x + y], [x_dot + y_dot]
jvp_rules[add_p] = add_jvp

def mul_jvp(primals, tangents):
  (x, y), (x_dot, y_dot) = primals, tangents
  return [x * y], [x_dot * y + x * y_dot]
jvp_rules[mul_p] = mul_jvp

def sin_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return [sin(x)], [cos(x) * x_dot]
jvp_rules[sin_p] = sin_jvp

def cos_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return [cos(x)], [-sin(x) * x_dot]
jvp_rules[cos_p] = cos_jvp

def neg_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return [neg(x)], [neg(x_dot)]
jvp_rules[neg_p] = neg_jvp

def reduce_sum_jvp(primals, tangents, *, axis):
  (x,), (x_dot,) = primals, tangents
  return [reduce_sum(x, axis)], [reduce_sum(x_dot, axis)]
jvp_rules[reduce_sum_p] = reduce_sum_jvp

def greater_jvp(primals, tangents):
  (x, y), _ = primals, tangents
  out_primal = greater(x, y)
  return [out_primal], [zeros_like(out_primal)]
jvp_rules[greater_p] = greater_jvp

def less_jvp(primals, tangents):
  (x, y), _ = primals, tangents
  out_primal = less(x, y)
  return [out_primal], [zeros_like(out_primal)]
jvp_rules[less_p] = less_jvp


# -

# Finally, we add a transformation API to kick off the trace:

def jvp_v1(f, primals, tangents):
  with new_main(JVPTrace) as main:
    trace = JVPTrace(main)
    tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
    out = f(*tracers_in)
    tracer_out = full_raise(trace, out)
    primal_out, tangent_out = tracer_out.primal, tracer_out.tangent
  return primal_out, tangent_out

# -

# ## Pytrees and flattening user functions' inputs and outputs

# A limitation with `jvp_v1` is that it assumes the user function accepts arrays
# as positional arguments and produces a single array as output. What if it
# produced a list as output? Or accepted nested containers as inputs? It would
# be a pain to deal with all the possible containers in inputs and outputs at
# every layer of the stack. Instead, we can wrap the user function so that the
# wrapped version accepts arrays as inputs and returns a flat list of arrays as
# output. The wrapper just needs to unflatten its input, call the user function,
# and flatten the output.
#
# Here's how we'd like to write `jvp`, assuming the user always gives us
# functions that take arrays as inputs and produces a flat list of arrays as
# outputs:

def jvp_flat(f, primals, tangents):
  with new_main(JVPTrace) as main:
    trace = JVPTrace(main)
    tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
    outs = f(*tracers_in)
    tracers_out = [full_raise(trace, out) for out in outs]
    primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)
  return primals_out, tangents_out


# To support user functions that have arbitrary containers in the inputs and
# outputs, here's how we'd write the user-facing `jvp` wrapper:

def jvp(f, primals, tangents):
  primals_flat, in_tree = tree_flatten(primals)
  tangents_flat, in_tree2 = tree_flatten(tangents)
  if in_tree != in_tree2: raise TypeError
  f, out_tree = flatten_fun(f, in_tree)
  primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
  primals_out = tree_unflatten(out_tree(), primals_out_flat)
  tangents_out = tree_unflatten(out_tree(), tangents_out_flat)
  return primals_out, tangents_out


# Notice that we had to plumb the tree structure of the user function output
# back to the caller of `flatten_fun`. That information isn't available until we
# actually run the user function, so `flatten_fun` just returns a reference to a
# mutable cell, represented as a thunk. These side-effects are safe because we
# always run the user function exactly once. (This safe regime is the reason for
# the "linear" name in `linear_util.py`, in the sense of [linear
# types](https://en.wikipedia.org/wiki/Substructural_type_system).)
#
# All that remains is to write `tree_flatten`, `tree_unflatten`, and
# `flatten_fun`.

# + tags=["hide-input"]
def flatten_fun(f, in_tree):
  store = Store()

  def flat_fun(*args_flat):
    pytree_args = tree_unflatten(in_tree, args_flat)
    out = f(*pytree_args)
    out_flat, out_tree = tree_flatten(out)
    store.set_value(out_tree)
    return out_flat

  return flat_fun, store

class Empty: pass
empty = Empty()

class Store:
  val = empty

  def set_value(self, val):
    assert self.val is empty
    self.val = val

  def __call__(self):
    return self.val


# + tags=["hide-input"]
import itertools as it
from typing import Callable, Type, Hashable, Dict, Iterable, Iterator

class NodeType(NamedTuple):
  name: str
  to_iterable: Callable
  from_iterable: Callable

def register_pytree_node(ty: Type, to_iter: Callable, from_iter: Callable
                         ) -> None:
  node_types[ty] = NodeType(str(ty), to_iter, from_iter)

node_types: Dict[Type, NodeType] = {}
register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
register_pytree_node(list,  lambda l: (None, l), lambda _, xs:  list(xs))
register_pytree_node(dict,
                     lambda d: map(tuple, unzip2(sorted(d.items()))),
                     lambda keys, vals: dict(zip(keys, vals)))

class PyTreeDef(NamedTuple):
  node_type: NodeType
  node_metadata: Hashable
  child_treedefs: Tuple['PyTreeDef']

class Leaf: pass
leaf = Leaf()

def tree_flatten(x: Any) -> Tuple[List[Any], PyTreeDef]:
  children_iter, treedef = _tree_flatten(x)
  return list(children_iter), treedef

def _tree_flatten(x: Any) -> Tuple[Iterable, PyTreeDef]:
  node_type = node_types.get(type(x))
  if node_type:
    node_metadata, children = node_type.to_iterable(x)
    children_flat, child_trees = unzip2(map(_tree_flatten, children))
    flattened = it.chain.from_iterable(children_flat)
    return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))
  else:
    return [x], leaf

def tree_unflatten(treedef: PyTreeDef, xs: List[Any]) -> Any:
  return _tree_unflatten(treedef, iter(xs))

def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> Any:
  if treedef is leaf:
    return next(xs)
  else:
    children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
    return treedef.node_type.from_iterable(treedef.node_metadata, children)


# -

# With this pytree-handling `jvp` implementation, we can now handle arbitrary
# input and output containers. That'll come in handy with future transformations
# too!

# # +
# def f(x):
#   y = sin(x) * 2.
#   z = - y + x
#   return {'hi': z, 'there': [x, y]}
#
# x, xdot = 3., 1.
# y, ydot = jvp(f, (x,), (xdot,))
# print(y)
# print(ydot)

# -

# ### Vectorized batching with `vmap`
#
# First, a couple helper functions, one for producing mapped abstract values
# from unmapped ones (by removing an axis), and one for moving batch dimensions
# around:

# +
def mapped_aval(batch_dim, aval):
  shape = list(aval.shape)
  del shape[batch_dim]
  return ShapedArray(tuple(shape), aval.dtype)

def move_batch_axis(axis_size, src, dst, x):
  if src is not_mapped:
    target_shape = list(np.shape(x))
    target_shape.insert(dst, axis_size)
    return broadcast(x, target_shape, [dst])
  elif src == dst:
    return x
  else:
    return moveaxis(x, src, dst)

def moveaxis(x, src: int, dst: int):
  perm = [i for i in range(np.ndim(x)) if i != src]
  perm.insert(dst, src)
  return transpose(x, perm)


# -

# The `Tracer` for vectorized batching carries a batched value and an optional
# integer indicating which axis (if any) is the batch axis.

# +
from typing import Union

class NotMapped: pass
not_mapped = NotMapped()

BatchAxis = Union[NotMapped, int]

class BatchTracer(Tracer):
  def __init__(self, trace, val, batch_dim: BatchAxis):
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim

  @property
  def aval(self):
    if self.batch_dim is not_mapped:
      return get_aval(self.val)
    else:
      return mapped_aval(self.batch_dim, get_aval(self.val))

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return full_lower(self.val)
    else:
      return self

class BatchTrace(Trace):
  pure = lift = lambda self, val: BatchTracer(self, val, not_mapped)

  def process_primitive(self, primitive, tracers, params):
    vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
    vmap_rule = vmap_rules[primitive]
    val_outs, bdim_outs = vmap_rule(self.axis_size, vals_in, bdims_in, **params)
    return [BatchTracer(self, x, bd) for x, bd in zip(val_outs, bdim_outs)]

  @property
  def axis_size(self):
    return self.main.global_data

vmap_rules = {}
# -

# Here we've implemented the optional `Tracer.full_lower` method, which lets us
# peel off a batching tracer if it's not needed because it doesn't represent a
# batched value.
#
# For `BatchTrace`, analogous to `JVPTrace`, the methods `pure` and `lift` just
# box a value in a `BatchTracer` with the minimal amount of context, which in
# this case is a `batch_dim` taking the sentinel value `not_mapped`. Notice we
# use the `MainTrace`'s interpreter-global data field to store the batch axis
# size.
#
# Next we can define batching interpreter rules for each primitive:

# +
from functools import partial

def binop_batching_rule(op, axis_size, vals_in, dims_in):
  (x, y), (x_bdim, y_bdim) = vals_in, dims_in
  if x_bdim != y_bdim:
    if x_bdim is not_mapped:
      x = move_batch_axis(axis_size, x_bdim, y_bdim, x)
      x_bdim = y_bdim
    else:
      y = move_batch_axis(axis_size, y_bdim, x_bdim, y)
  return [op(x, y)], [x_bdim]
vmap_rules[add_p] = partial(binop_batching_rule, add)
vmap_rules[mul_p] = partial(binop_batching_rule, mul)

def vectorized_unop_batching_rule(op, axis_size, vals_in, dims_in):
  (x,), (x_bdim,) = vals_in, dims_in
  return [op(x)], [x_bdim]
vmap_rules[sin_p] = partial(vectorized_unop_batching_rule, sin)
vmap_rules[cos_p] = partial(vectorized_unop_batching_rule, cos)
vmap_rules[neg_p] = partial(vectorized_unop_batching_rule, neg)

def reduce_sum_batching_rule(axis_size, vals_in, dims_in, *, axis):
  (x,), (x_bdim,) = vals_in, dims_in
  new_axis = axis + (x_bdim <= axis)
  out_bdim = x_bdim - (new_axis < x_bdim)
  return [reduce_sum(x, new_axis)], [out_bdim]
vmap_rules[reduce_sum_p] = reduce_sum_batching_rule


# -

# Finally, we add a transformation API to kick off the trace:

# +
def vmap_flat(f, in_axes, *args):
  axis_size, = {x.shape[ax] for x, ax in zip(args, in_axes)
                if ax is not not_mapped}
  with new_main(BatchTrace, axis_size) as main:
    trace = BatchTrace(main)
    tracers_in = [BatchTracer(trace, x, ax) if ax is not None else x
                  for x, ax in zip(args, in_axes)]
    outs = f(*tracers_in)
    tracers_out = [full_raise(trace, out) for out in outs]
    vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
  outs_transposed = [move_batch_axis(axis_size, bdim, 0, val_out)
                     for val_out, bdim in zip(vals_out, bdims_out)]
  return outs_transposed

def vmap(f, in_axes):
  def batched_f(*args):
    args_flat, in_tree = tree_flatten(args)
    in_axes_flat, in_tree2 = tree_flatten(in_axes)
    if in_tree != in_tree2: raise TypeError
    f_flat, out_tree = flatten_fun(f, in_tree)
    outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
    return tree_unflatten(out_tree(), outs_flat)
  return batched_f


# +
def add_one_to_a_scalar(scalar):
  assert np.ndim(scalar) == 0
  return 1 + scalar

vector_in = np.arange(3.)
vector_out = vmap(add_one_to_a_scalar, (0,))(vector_in)

print(vector_in)
print(vector_out)


# +
def jacfwd(f, x):
  pushfwd = lambda v: jvp(f, (x,), (v,))[1]
  vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
  return vmap(pushfwd, (0,))(vecs_in)

def f(x):
  return sin(x)

jacfwd(f, np.arange(3.))
# -

# That's it for `jvp` and `vmap`!


# ## Part 2: Jaxprs
#
# The next transformations on the horizon are `jit` for just-in-time
# compilation and `vjp` for reverse-mode autodiff.  (`grad` is just a small
# wrapper around `vjp`.) Whereas `jvp` and `vmap` only needed each `Tracer` to
# carry a little bit of extra context, for both `jit` and `vjp` we need much
# richer context: we need to represent _programs_. That is, we need jaxprs!
#
# Jaxprs are JAX's internal intermediate representation of programs. They are
# explicitly typed, functional, first-order, and in ANF form. We need a
# program representation for `jit` because the purpose of `jit` is to stage
# computation out of Python. For any computation we want to stage out, we need
# to be able to represent it as data, and build it up as we trace a Python
# function. Similarly, `vjp` needs a way to represent the computation for the
# backward pass of reverse-mode autodiff. We use the same jaxpr program
# representation for both needs.
#
# (Building a program representation is the most
# [free](https://en.wikipedia.org/wiki/Free_object) kind of
# trace-transformation, and so except for issues around handling native Python
# control flow, any transformation could be implemented by first tracing to a
# jaxpr and then interpreting the jaxpr.)

# ### Jaxpr data structures
#
# The jaxpr term syntax is roughly:
#
# ```
# jaxpr ::=
#   { lambda <binder> , ... .
#     let <eqn>
#         ...
#     in ( <atom> , ... ) }
#
# binder ::= <var>:<array_type>
# var ::= a | b | c | ...
# atom ::= <var> | <literal>
# literal ::= <int32> | <int64> | <float32> | <float64>
#
# eqn ::= <binder> , ... = <primitive> [ <params> ] <atom> , ...
# ```
#
# The syntax of types is:
#
# ```
# jaxpr_type ::= [ <array_type> , ... ] -> [ <array_type> , ... ]
# array_type ::= <dtype>[<shape>]
# dtype ::= f32 | f64 | i32 | i64
# shape ::= <int> , ...
# ```
#
# How do we represent these as Python data structures? We reuse ShapedArrays to
# represent types, and we can represent the term syntax with a few Python
# structs:

# +
from typing import Set

class Var:
  aval: ShapedArray
  def __init__(self, aval): self.aval = aval

class Lit:
  val: Any
  aval: ShapedArray

  def __init__(self, val):
    self.aval = aval = raise_to_shaped(get_aval(val))
    self.val = np.array(val, aval.dtype)

Atom = Union[Var, Lit]

class JaxprEqn(NamedTuple):
  primitive: Primitive
  inputs: List[Atom]
  params: Dict[str, Any]
  out_binders: List[Var]

class Jaxpr(NamedTuple):
  in_binders: List[Var]
  eqns: List[JaxprEqn]
  outs: List[Atom]

  def __hash__(self): return id(self)
  __eq__ = op.is_

def raise_to_shaped(aval):
  return ShapedArray(aval.shape, aval.dtype)


# -

# Type-checking a jaxpr involves checking that there are no unbound variables,
# that variables are only bound once, and that for each equation the type of
# the primitive application matches the type of the output binders.

# +
class JaxprType(NamedTuple):
  in_types:  List[ShapedArray]
  out_types: List[ShapedArray]

  def __repr__(self):
    in_types = ', '.join(aval.str_short() for aval in self.in_types)
    out_types = ', '.join(aval.str_short() for aval in self.out_types)
    return f'({in_types}) -> ({out_types})'

def typecheck_jaxpr(jaxpr: Jaxpr) -> JaxprType:
  env: Set[Var] = set()

  for v in jaxpr.in_binders:
    if v in env: raise TypeError
    env.add(v)

  for eqn in jaxpr.eqns:
    in_types = [typecheck_atom(env, x) for x in eqn.inputs]
    out_types = abstract_eval_rules[eqn.primitive](*in_types, **eqn.params)
    for out_binder, out_type in zip(eqn.out_binders, out_types):
      if not out_type == out_binder.aval: raise TypeError
    for out_binder in eqn.out_binders:
      if out_binder in env: raise TypeError
      env.add(out_binder)

  in_types = [v.aval for v in jaxpr.in_binders]
  out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
  return JaxprType(in_types, out_types)

def typecheck_atom(env: Set[Var], x: Atom) -> ShapedArray:
  if isinstance(x, Var):
    if x not in env: raise TypeError("unbound variable")
    return x.aval
  elif isinstance(x, Lit):
    return raise_to_shaped(get_aval(x.val))
  else:
    assert False


# -

# We can apply the function represented by a jaxpr to arguments with a simple
# interpreter.

# +
def eval_jaxpr(jaxpr: Jaxpr, args: List[Any]) -> List[Any]:
  env: Dict[Var, Any] = {}

  def read(x: Atom) -> Any:
    return env[x] if type(x) is Var else x.val

  def write(v: Var, val: Any) -> None:
    assert v not in env  # single-assignment
    env[v] = val

  map(write, jaxpr.in_binders, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.inputs)
    outs = bind(eqn.primitive, *in_vals, **eqn.params)
    map(write, eqn.out_binders, outs)
  return map(read, jaxpr.outs)

def jaxpr_as_fun(jaxpr: Jaxpr):
  return lambda *args: eval_jaxpr(jaxpr, args)


# -

# By using `bind` in the interpreter, this interpreter itself is traceable.

# ### Building jaxprs with tracing
#
# Now that we have jaxprs as a data structure, we need ways to produce these
# from tracing Python code. In general there are two variants of how we trace to
# a jaxpr; `jit` uses one and `vjp` uses the other. We'll start with the one
# used by `jit`, which is also used by control flow primitives like `lax.cond`,
# `lax.while_loop`, and `lax.scan`.

# +
def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
  assert 0 <= n <= len(lst)
  return lst[:n], lst[n:]

def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
  assert len(bs) == len(l)
  lists = lst1, lst2 = [], []
  for b, x in zip(bs, l):
    lists[b].append(x)
  return lst1, lst2


# +
# NB: the analogous class in JAX is called 'DynamicJaxprTracer'
class JaxprTracer(Tracer):
  __slots__ = ['aval']
  aval: ShapedArray

  def __init__(self, trace, aval):
    self._trace = trace
    self.aval = aval

# NB: the analogous class in JAX is called 'DynamicJaxprTrace'
class JaxprTrace(Trace):
  def new_arg(self, aval: ShapedArray) -> JaxprTracer:
    aval = raise_to_shaped(aval)
    tracer = self.builder.new_tracer(self, aval)
    self.builder.tracer_to_var[id(tracer)] = Var(aval)
    return tracer

  def get_or_make_const_tracer(self, val: Any) -> JaxprTracer:
    tracer = self.builder.const_tracers.get(id(val))
    if tracer is None:
      tracer = self.builder.new_tracer(self, raise_to_shaped(get_aval(val)))
      self.builder.add_const(tracer, val)
    return tracer
  pure = lift = get_or_make_const_tracer

  def process_primitive(self, primitive, tracers, params):
    avals_in = [t.aval for t in tracers]
    avals_out = abstract_eval_rules[primitive](*avals_in, **params)
    out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
    inputs = [self.builder.getvar(t) for t in tracers]
    outvars = [self.builder.add_var(t) for t in out_tracers]
    self.builder.add_eqn(JaxprEqn(primitive, inputs, params, outvars))
    return out_tracers

  @property
  def builder(self):
    return self.main.global_data

# NB: in JAX, we instead attach abstract eval rules to Primitive instances
abstract_eval_rules = {}


# -

# Notice that we keep as interpreter-global data a builder object, which keeps
# track of variables, constants, and eqns as we build up the jaxpr.

class JaxprBuilder:
  eqns: List[JaxprEqn]
  tracer_to_var: Dict[int, Var]
  const_tracers: Dict[int, JaxprTracer]
  constvals: Dict[Var, Any]
  tracers: List[JaxprTracer]

  def __init__(self):
    self.eqns = []
    self.tracer_to_var = {}
    self.const_tracers = {}
    self.constvals = {}
    self.tracers = []

  def new_tracer(self, trace: JaxprTrace, aval: ShapedArray) -> JaxprTracer:
    tracer = JaxprTracer(trace, aval)
    self.tracers.append(tracer)
    return tracer

  def add_eqn(self, eqn: JaxprEqn) -> None:
    self.eqns.append(eqn)

  def add_var(self, tracer: JaxprTracer) -> Var:
    assert id(tracer) not in self.tracer_to_var
    var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
    return var

  def getvar(self, tracer: JaxprTracer) -> Var:
    var = self.tracer_to_var.get(id(tracer))
    assert var is not None
    return var

  def add_const(self, tracer: JaxprTracer, val: Any) -> Var:
    var = self.add_var(tracer)
    self.const_tracers[id(val)] = tracer
    self.constvals[var] = val
    return var

  def build(self, in_tracers: List[JaxprTracer], out_tracers: List[JaxprTracer]
            ) -> Tuple[Jaxpr, List[Any]]:
    constvars, constvals = unzip2(self.constvals.items())
    t2v = lambda t: self.tracer_to_var[id(t)]
    in_binders = constvars + [t2v(t) for t in in_tracers]
    out_vars = [t2v(t) for t in out_tracers]
    jaxpr = Jaxpr(in_binders, self.eqns, out_vars)
    typecheck_jaxpr(jaxpr)
    jaxpr, constvals = _inline_literals(jaxpr, constvals)
    return jaxpr, constvals


def _inline_literals(jaxpr: Jaxpr, consts: List[Any]) -> Tuple[Jaxpr, List[Any]]:
  const_binders, other_binders = split_list(jaxpr.in_binders, len(consts))
  scalars = [type(x) in jax_types and not get_aval(x).shape for x in consts]
  new_const_binders, lit_binders = partition_list(scalars, const_binders)
  new_consts, lit_vals = partition_list(scalars, consts)
  literals = dict(zip(lit_binders, map(Lit, lit_vals)))
  new_eqns = [JaxprEqn(eqn.primitive, [literals.get(x, x) for x in eqn.inputs],
                       eqn.params, eqn.out_binders) for eqn in jaxpr.eqns]
  new_outs = [literals.get(x, x) for x in jaxpr.outs]
  new_jaxpr = Jaxpr(new_const_binders + other_binders, new_eqns, new_outs)
  typecheck_jaxpr(new_jaxpr)
  return new_jaxpr, new_consts


# The rules we need for `JaxprTrace.process_primitive` are essentially typing
# rules for primitive applications: given the primitive, its parameters, and
# types for the inputs, the rule must produce a type for the output, which is
# then packaged with the output `JaxprTracer`. We can use abstract evaluation
# rules for this same purpose, even though they can be more general (since
# abstract evaluation rules must accept ConcreteArray inputs, and since they
# need only return an upper bound on the set of possible outputs, they can
# produce ConcreteArray outputs as well). We'll reuse these abstract evaluation
# rules for the other jaxpr-producing trace machinery, where the potential extra
# generality is useful.

# +
def binop_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
  if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
    raise TypeError
  if raise_to_shaped(x) != raise_to_shaped(y): raise TypeError
  return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[add_p] = binop_abstract_eval
abstract_eval_rules[mul_p] = binop_abstract_eval

def compare_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
  if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
    raise TypeError
  if x.shape != y.shape: raise TypeError
  return [ShapedArray(x.shape, np.dtype('bool'))]
abstract_eval_rules[greater_p] = compare_abstract_eval
abstract_eval_rules[less_p] = compare_abstract_eval

def vectorized_unop_abstract_eval(x: ShapedArray) -> List[ShapedArray]:
  return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[sin_p] = vectorized_unop_abstract_eval
abstract_eval_rules[cos_p] = vectorized_unop_abstract_eval
abstract_eval_rules[neg_p] = vectorized_unop_abstract_eval

def reduce_sum_abstract_eval(x: ShapedArray, *, axis: int) -> List[ShapedArray]:
  new_shape = [d for i, d in enumerate(x.shape) if i != axis]
  return [ShapedArray(tuple(new_shape), x.dtype)]
abstract_eval_rules[reduce_sum_p] = reduce_sum_abstract_eval

def broadcast_abstract_eval(x: ShapedArray, *, shape: Sequence[int],
                            axes: Sequence[int]) -> List[ShapedArray]:
  return [ShapedArray(tuple(shape), x.dtype)]
abstract_eval_rules[broadcast_p] = broadcast_abstract_eval
# -

# To check our implementation of jaxprs, we can add a `make_jaxpr`
# transformation and a pretty-printer:

# +
from functools import lru_cache

@lru_cache()  # ShapedArrays are hashable
def make_jaxpr_v1(f, *avals_in):
  avals_in, in_tree = tree_flatten(avals_in)
  f, out_tree = flatten_fun(f, in_tree)

  builder = JaxprBuilder()
  with new_main(JaxprTrace, builder) as main:
    trace = JaxprTrace(main)
    tracers_in = [trace.new_arg(aval) for aval in avals_in]
    outs = f(*tracers_in)
    tracers_out = [full_raise(trace, out) for out in outs]
    jaxpr, consts = builder.build(tracers_in, tracers_out)
  return jaxpr, consts, out_tree()


# + tags=["hide-input"]
from typing import DefaultDict
from collections import defaultdict
import string

class PPrint:
  lines: List[Tuple[int, str]]

  def __init__(self, lines):
    self.lines = lines

  def indent(self, indent: int) -> 'PPrint':
    return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

  def __add__(self, rhs: 'PPrint') -> 'PPrint':
    return PPrint(self.lines + rhs.lines)

  def __rshift__(self, rhs: 'PPrint') -> 'PPrint':
    if not rhs.lines: return self
    if not self.lines: return rhs
    indent, s = self.lines[-1]
    indented_block = rhs.indent(indent + len(s))
    common_line = s + ' ' * rhs.lines[0][0] + rhs.lines[0][1]
    return PPrint(self.lines[:-1]
                  + [(indent, common_line)]
                  + indented_block.lines[1:])

  def __str__(self) -> str:
    return '\n'.join(' ' * indent + s for indent, s in self.lines)

def pp(s: Any) -> PPrint:
  return PPrint([(0, line) for line in str(s).splitlines()])

def vcat(ps: List[PPrint]) -> PPrint:
  return sum(ps, pp(''))

def pp_jaxpr(jaxpr: Jaxpr) -> PPrint:
  namegen = (''.join(s) for r in it.count(1)
             for s in it.permutations(string.ascii_lowercase, r))
  names = defaultdict(lambda: next(namegen))
  in_binders = ', '.join(var_str(names, x) for x in jaxpr.in_binders)
  eqns = vcat([pp_eqn(names, e) for e in jaxpr.eqns])
  outs = ', '.join(names[v] if isinstance(v, Var) else str(v.val)
                   for v in jaxpr.outs)
  return (pp(f'{{ lambda {in_binders} .') +
          ((pp('let ') >> eqns) + pp(f'in ( {outs} ) }}')).indent(2))

def var_str(names: DefaultDict[Var, str], v: Var) -> str:
  return f'{names[v]}:{v.aval.str_short()}'

def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
  rule = pp_rules.get(eqn.primitive)
  if rule:
    return rule(names, eqn)
  else:
    lhs = pp(' '.join(var_str(names, v) for v in eqn.out_binders))
    rhs = (pp(eqn.primitive.name) >> pp_params(eqn.params) >>
           pp(' '.join(names[x] if isinstance(x, Var) else str(x.val)
                       for x in eqn.inputs)))
    return lhs >> pp(' = ') >> rhs

def pp_params(params: Dict[str, Any]) -> PPrint:
  items = sorted(params.items())
  if items:
    return pp(' [ ') >> vcat([pp(f'{k}={v}') for k, v in items]) >> pp(' ] ')
  else:
    return pp(' ')

Jaxpr.__repr__ = lambda self: str(pp_jaxpr(self))
pp_rules: Dict[Primitive, Callable[..., PPrint]] = {}
# -

jaxpr, consts, _ = make_jaxpr_v1(lambda x: 2. * x, raise_to_shaped(get_aval(3.)))
print(jaxpr)
print(typecheck_jaxpr(jaxpr))

# But there's a limitation here: because of how `find_top_trace` operates by
# data dependence, `make_jaxpr_v1` can't stage out all the primitive operations
# performed by the Python callable it's given. For example:

jaxpr, consts, _ = make_jaxpr_v1(lambda: mul(2., 2.))
print(jaxpr)


# This is precisely the issue that
# [omnistaging](https://github.com/google/jax/pull/3370) fixed.
# We want to ensure that the `JaxprTrace` started by `make_jaxpr` is always
# applied, regardless of whether any inputs to `bind` are boxed in corresponding
# `JaxprTracer` instances. We can achieve this by employing the `dynamic_trace`
# global defined in Part 1:

# +
@contextmanager
def new_dynamic(main: MainTrace):
  global dynamic_trace
  prev_dynamic_trace, dynamic_trace = dynamic_trace, main
  try:
    yield
  finally:
    dynamic_trace = prev_dynamic_trace

@lru_cache()
def make_jaxpr(f: Callable, *avals_in: ShapedArray,
               ) -> Tuple[Jaxpr, List[Any], PyTreeDef]:
  avals_in, in_tree = tree_flatten(avals_in)
  f, out_tree = flatten_fun(f, in_tree)

  builder = JaxprBuilder()
  with new_main(JaxprTrace, builder) as main:
    with new_dynamic(main):
      trace = JaxprTrace(main)
      tracers_in = [trace.new_arg(aval) for aval in avals_in]
      outs = f(*tracers_in)
      tracers_out = [full_raise(trace, out) for out in outs]
      jaxpr, consts = builder.build(tracers_in, tracers_out)
  return jaxpr, consts, out_tree()

jaxpr, consts, _ = make_jaxpr(lambda: mul(2., 2.))
print(jaxpr)

# -

# Using `dynamic_trace` this way is conceptually the same as stashing the
# current interpreter stack and starting a new one with the `JaxprTrace` at the
# bottom. That is, no interpreters lower in the stack than the `dynamic_trace`
# are applied (since `JaxprTrace.process_primitive` doesn't call `bind`), though
# if the Python callable being traced to a jaxpr itself uses transformations
# then those can be pushed onto the interpreter stack above the `JaxprTrace`.
# But temporarily stashing the interpreter stack would break up the system
# state. The `dynamic_trace` tag achieves the same goals while keeping the
# system state simpler.

# That's it for jaxprs! With jaxprs in hand, we can implement the remaining
# major JAX features.

# ## Part 3: `jit`, simplified
#
# While `jit` has a transformation-like API in that it accepts a Python callable
# as an argument, under the hood it's really a higher-order primitive rather
# than a transformation. A primitive is _higher-order_ when it's parameterized
# by a function.

# ### On-the-fly ("final style") and staged ("initial style") processing
#
# There are two options for how to handle higher-order primitives. Each requires
# a different approach to tracing and engenders different tradeoffs:
# 1. **On-the-fly processing, where `bind` takes a Python callable as an
#    argument.** We defer forming a jaxpr until as late as possible, namely
#    until we're running the final interpreter at the bottom of the interpreter
#    stack. That way we can swap a `JaxprTrace` in at the bottom of the
#    interpreter stack and thus stage out rather than execute all primitive
#    operations. With this approach, transformations in the stack get applied as
#    we execute the Python callable as usual. This approach can be very tricky
#    to implement, but it's as general as possible because it allows
#    higher-order primitives not to raise the abstraction level of their
#    arguments and thus allows data-dependent Python control flow. We refer to
#    this approach as using a "final-style higher-order primitive" employing the
#    discharge-at-tracing-time "final-style transformations" we've used so far.
# 2. **Staged processing, where `bind` takes a jaxpr as an argument.** Before we
#    call `bind`, in the primitive wrapper we can just use `make_jaxpr` to form
#    a jaxpr up-front and be done with the Python callable entirely. In this
#    case, `make_jaxpr` puts its `JaxprTrace` at the top of the interpreter
#    stack, and no transformations lower in the stack, which might enter via
#    closed-over Tracers, are applied to the Python callable as we trace it.
#    (Transformations applied within the Python callable are applied as usual,
#    being added to the stack above the JaxprTrace.) Instead, the
#    transformations lower in the stack are later applied to the call primitive,
#    and the call primitive's rules must then transform the jaxpr itself.
#    Because we trace to a jaxpr up-front, this approach can't support
#    data-dependent Python control flow, but it is more straightforward to
#    implement. We refer to this kind of higher-order primitive as an
#    "initial-style higher-order primitive", and say that its jaxpr-processing
#    transformation rules are "initial-style transformation rules."
#
# The latter approach fits for `jit` because we don't need to support
# data-dependent Python control flow in the user-provided Python callable, as
# the whole purpose of `jit` is to stage computation out of Python to be
# executed by XLA. (In contrast, `custom_jvp` is a higher-order primitive in
# which we want to support data-dependent Python control flow.)
#
# Historically, we started using the "initial-style" and "final-style"
# terminology after reading the [typed tagless final
# interpreters](http://okmij.org/ftp/tagless-final/index.html) paper, and
# jokingly referring to JAX as an implementation of "untyped tagful final
# interpreters." We don't claim to carry over (or understand) any deep meaning
# behind these terms; we loosely use "initial style" to mean "build an AST and
# then transform it", and we use "final style" to mean "transform as we trace."
# But it's just imprecise yet sticky jargon.

# With the initial-style approach, here's the user-facing `jit` wrapper:

# +
def jit(f):
  def f_jitted(*args):
    avals_in = [raise_to_shaped(get_aval(x)) for x in args]
    jaxpr, consts, out_tree = make_jaxpr(f, *avals_in)
    outs = bind(xla_call_p, *consts, *args, jaxpr=jaxpr, num_consts=len(consts))
    out = tree_unflatten(out_tree, outs)
    # breakpoint()
    return out
  return f_jitted

xla_call_p = Primitive('xla_call')


# -

# With any new primitive, we need to give it transformation rules, starting with
# its evaluation rule. When we evaluate an application of the `xla_call`
# primitive, we want to stage out out the computation to XLA. That involves
# translating the jaxpr to an XLA HLO program, transferring the argument values
# to the XLA device, executing the XLA program, and transferring back the
# results. We'll cache the XLA HLO compilation so that for each `jit`ted
# function it only needs to be performed once per argument shape and dtype
# signature.
#
# First, some utilities.

class IDHashable:
  val: Any

  def __init__(self, val):
    self.val = val

  def __hash__(self) -> int:
    return id(self.val)

  def __eq__(self, other):
    return type(other) is IDHashable and id(self.val) == id(other.val)

# Next, we'll define the evaluation rule for `xla_call`:

# +
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
xe = xc._xla
xops = xc._xla.ops

def xla_call_impl(*args, jaxpr: Jaxpr, num_consts: int):
  # consts, args = args[:num_consts], args[num_consts:]
  # hashable_consts = tuple(map(IDHashable, consts))
  # execute = xla_callable(IDHashable(jaxpr), hashable_consts)
  execute = jaxpr_as_fun(jaxpr)
  out = execute(*args)
  return out
impl_rules[xla_call_p] = xla_call_impl

# @lru_cache()
# def xla_callable(hashable_jaxpr: IDHashable, hashable_consts: Tuple[IDHashable]):
#   jaxpr: Jaxpr = hashable_jaxpr.val
#   typecheck_jaxpr(jaxpr)
#   consts = [x.val for x in hashable_consts]
#   in_avals = [v.aval for v in jaxpr.in_binders[len(consts):]]
#   c = xc.XlaBuilder('xla_call')
#   xla_consts = _xla_consts(c, consts)
#   xla_params = _xla_params(c, in_avals)
#   outs = jaxpr_subcomp(c, jaxpr, xla_consts + xla_params)
#   out = xops.Tuple(c, outs)
#   compiled = xb.get_backend(None).compile(c.build(out))
#   return partial(execute_compiled, compiled, [v.aval for v in jaxpr.outs])
#
# def _xla_consts(c: xe.XlaBuilder, consts: List[Any]) -> List[xe.XlaOp]:
#   unique_consts = {id(cnst): cnst for cnst in consts}
#   xla_consts = {
#       id_: xops.ConstantLiteral(c, cnst) for id_, cnst in unique_consts.items()}
#   return [xla_consts[id(cnst)] for cnst in consts]
#
# def _xla_params(c: xe.XlaBuilder, avals_in: List[ShapedArray]) -> List[xe.XlaOp]:
#   return [xops.Parameter(c, i, _xla_shape(a)) for i, a in enumerate(avals_in)]
#
# def _xla_shape(aval: ShapedArray) -> xe.Shape:
#   return xc.Shape.array_shape(xc.dtype_to_etype(aval.dtype), aval.shape)


# -

# The main action is in `xla_callable`, which compiles a jaxpr into an XLA HLO
# program using `jaxpr_subcomp`, then returns a callable which executes the
# compiled program:

# +
def jaxpr_subcomp(c: xe.XlaBuilder, jaxpr: Jaxpr, args: List[xe.XlaOp]
                  ) -> xe.XlaOp:
  env: Dict[Var, xe.XlaOp] = {}

  def read(x: Atom) -> xe.XlaOp:
    return env[x] if type(x) is Var else xops.Constant(c, np.asarray(x.val))

  def write(v: Var, val: xe.XlaOp) -> None:
    env[v] = val

  map(write, jaxpr.in_binders, args)
  for eqn in jaxpr.eqns:
    in_avals = [x.aval for x in eqn.inputs]
    in_vals = map(read, eqn.inputs)
    rule = xla_translations[eqn.primitive]
    out_vals = rule(c, in_avals, in_vals, **eqn.params)
    map(write, eqn.out_binders, out_vals)
  return map(read, jaxpr.outs)

def execute_compiled(compiled, out_avals, *args):
  input_bufs = [input_handlers[type(x)](x) for x in args]
  out_bufs = compiled.execute(input_bufs)
  return [handle_result(aval, buf) for aval, buf in zip(out_avals, out_bufs)]

default_input_handler = xb.get_backend(None).buffer_from_pyval
input_handlers = {ty: default_input_handler for ty in
                  [bool, int, float, np.ndarray, np.float64, np.float32]}

def handle_result(aval: ShapedArray, buf):
  del aval  # Unused for now
  return buf.to_py()

xla_translations = {}


# -

# Notice that `jaxpr_subcomp` has the structure of a simple interpreter. That's
# a common pattern: the way we process jaxprs is usually with an interpreter.
# And as with any interpreter, we need an interpretation rule for each
# primitive:

# +
def direct_translation(op, c, in_avals, in_vals):
  del c, in_avals
  return [op(*in_vals)]

xla_translations[add_p] = partial(direct_translation, xops.Add)
xla_translations[mul_p] = partial(direct_translation, xops.Mul)
xla_translations[neg_p] = partial(direct_translation, xops.Neg)
xla_translations[sin_p] = partial(direct_translation, xops.Sin)
xla_translations[cos_p] = partial(direct_translation, xops.Cos)
xla_translations[greater_p] = partial(direct_translation, xops.Gt)
xla_translations[less_p] = partial(direct_translation, xops.Lt)

def reduce_sum_translation(c, in_avals, in_vals, *, axis):
  (x_aval,), (x,) = in_avals, in_vals
  zero = xops.ConstantLiteral(c, np.array(0, x_aval.dtype))
  subc = xc.XlaBuilder('add')
  shape = _xla_shape(ShapedArray((), x_aval.dtype))
  xops.Add(xops.Parameter(subc, 0, shape), xops.Parameter(subc, 1, shape))
  return [xops.Reduce(c, [x], [zero], subc.build(), [axis])]
xla_translations[reduce_sum_p] = reduce_sum_translation

def broadcast_translation(c, in_avals, in_vals, *, shape, axes):
  x, = in_vals
  dims_complement = [i for i in range(len(shape)) if i not in axes]
  return [xops.BroadcastInDim(x, shape, dims_complement)]
xla_translations[broadcast_p] = broadcast_translation


# -

# With that, we can now use `jit` to stage out, compile, and execute programs
# with XLA!

@jit
def f(x, y):
  print('tracing!')
  return sin(x) * cos(y)


z = f(3., 4.)  # 'tracing!' prints the first time
print(z)

z = f(4., 5.)  # 'tracing!' doesn't print, compilation cache hit!
print(z)


# +
@jit
def f(x):
  return reduce_sum(x, axis=0)

print(f(np.array([1., 2., 3.])))


# +
def f(x):
  y = sin(x) * 2.
  z = - y + x
  return z

def deriv(f):
  return lambda x: jvp(f, (x,), (1.,))[1]

print(    deriv(deriv(f))(3.))
print(jit(deriv(deriv(f)))(3.))


# -

# Instead of implementing `jit` to first trace to a jaxpr and then to lower the
# jaxpr to XLA HLO, it might appear that we could have skipped the jaxpr step
# and just lowered to HLO while tracing. That is, perhaps we could have instead
# implemented `jit` with a `Trace` and `Tracer` that appended to the XLA HLO
# graph incrementally on each primitive bind. That's correct for now, but won't
# be possible when we introduce compiled SPMD computations because there we must
# know the number of replicas needed before compiling the program.

# We haven't yet defined any transformation rules for `xla_call_p` other than
# its evaluation rule. That is, we can't yet do `vmap`-of-`jit` or
# `jvp`-of-`jit` or even `jit`-of`-jit`. Instead `jit` has to be at the "top
# level." Let's fix that!

# +
def xla_call_jvp_rule(primals, tangents, *, jaxpr, num_consts):
  del num_consts  # Unused
  new_jaxpr, new_consts = jvp_jaxpr(jaxpr)
  outs = bind(xla_call_p, *new_consts, *primals, *tangents, jaxpr=new_jaxpr,
              num_consts=len(new_consts))
  n = len(outs) // 2
  primals_out, tangents_out = outs[:n], outs[n:]
  return primals_out, tangents_out
jvp_rules[xla_call_p] = xla_call_jvp_rule

@lru_cache()
def jvp_jaxpr(jaxpr: Jaxpr) -> Tuple[Jaxpr, List[Any]]:
  def jvp_traceable(*primals_and_tangents):
    n = len(primals_and_tangents) // 2
    primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
    return jvp(jaxpr_as_fun(jaxpr), primals, tangents)

  in_avals = [v.aval for v in jaxpr.in_binders]
  new_jaxpr, new_consts, _ = make_jaxpr(jvp_traceable, *in_avals, *in_avals)
  return new_jaxpr, new_consts


# +
def xla_call_vmap_rule(axis_size, vals_in, dims_in, *, jaxpr, num_consts):
  del num_consts  # Unused
  new_jaxpr, new_consts = vmap_jaxpr(jaxpr, axis_size, tuple(dims_in))
  outs = bind(xla_call_p, *new_consts, *vals_in, jaxpr=new_jaxpr,
              num_consts=len(new_consts))
  return outs, [0] * len(outs)
vmap_rules[xla_call_p] = xla_call_vmap_rule

@lru_cache()
def vmap_jaxpr(jaxpr: Jaxpr, axis_size: int, bdims_in: Tuple[BatchAxis, ...]
               ) -> Tuple[Jaxpr, List[Any]]:
  vmap_traceable = vmap(jaxpr_as_fun(jaxpr), tuple(bdims_in))
  in_avals = [unmapped_aval(axis_size, d, v.aval)
              for v, d in zip(jaxpr.in_binders, bdims_in)]
  new_jaxpr, new_consts, _ = make_jaxpr(vmap_traceable, *in_avals)
  return new_jaxpr, new_consts

def unmapped_aval(axis_size: int, batch_dim: BatchAxis, aval: ShapedArray
                  ) -> ShapedArray:
  if batch_dim is not_mapped:
    return aval
  else:
    shape = list(aval.shape)
    shape.insert(batch_dim, axis_size)
    return ShapedArray(tuple(shape), aval.dtype)


# +
def xla_call_abstract_eval_rule(*in_types, jaxpr, num_consts):
  del num_consts  # Unused
  jaxpr_type = typecheck_jaxpr(jaxpr)
  if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
    raise TypeError
  return jaxpr_type.out_types
abstract_eval_rules[xla_call_p] = xla_call_abstract_eval_rule

def xla_call_translation(c, in_avals, in_vals, *, jaxpr, num_consts):
  del num_consts  # Only used at top-level.
  # Calling jaxpr_subcomp directly would inline. We generate a Call HLO instead.
  subc = xc.XlaBuilder('inner xla_call')
  xla_params = _xla_params(subc, in_avals)
  outs = jaxpr_subcomp(subc, jaxpr, xla_params)
  subc = subc.build(xops.Tuple(subc, outs))
  return destructure_tuple(c, xops.Call(c, subc, in_vals))
xla_translations[xla_call_p] = xla_call_translation

def destructure_tuple(c, tup):
  num_elements = len(c.get_shape(tup).tuple_shapes())
  return [xops.GetTupleElement(tup, i) for i in range(num_elements)]


# +
@jit
def f(x):
  print('tracing!')
  y = sin(x) * 2.
  z = - y + x
  return z

x, xdot = 3., 1.
y, ydot = jvp(f, (x,), (xdot,))
print(y)
print(ydot)
# -

y, ydot = jvp(f, (x,), (xdot,))  # 'tracing!' not printed

ys = vmap(f, (0,))(np.arange(3.))
print(ys)


# One piece missing is device memory persistence for arrays. That is, we've
# defined `handle_result` to transfer results back to CPU memory as NumPy
# arrays, but it's often preferable to avoid transferring results just to
# transfer them back for the next operation. We can do that by introducing a
# `DeviceArray` class, which can wrap XLA buffers and otherwise duck-type
# `numpy.ndarray`s:

# +
def handle_result(aval: ShapedArray, buf):  # noqa: F811
  return DeviceArray(aval, buf)

class DeviceArray:
  buf: Any
  aval: ShapedArray

  def __init__(self, aval, buf):
    self.aval = aval
    self.buf = buf

  dtype = property(lambda self: self.aval.dtype)
  shape = property(lambda self: self.aval.shape)
  ndim  = property(lambda self: self.aval.ndim)

  def __array__(self): return self.buf.to_py()
  def __repr__(self):  return repr(self.buf.to_py())
  def __str__(self):   return str(self.buf.to_py())

  _neg = staticmethod(neg)
  _add = staticmethod(add)
  _radd = staticmethod(add)
  _mul = staticmethod(mul)
  _rmul = staticmethod(mul)
  _gt = staticmethod(greater)
  _lt = staticmethod(less)
input_handlers[DeviceArray] = lambda x: x.buf

jax_types.add(DeviceArray)


# +
@jit
def f(x):
  y = sin(x) * 2.
  z = - y + x
  return z

x, xdot = 3., 1.
y, ydot = jvp(f, (x,), (xdot,))
print(y)
print(ydot)


# + tags=["hide-input"]
def pprint_xla_call(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
  lhs = pp(' '.join(var_str(names, v) for v in eqn.out_binders))
  params_without_jaxpr = {k:v for k, v in eqn.params.items() if k != 'jaxpr'}
  rhs = (pp(eqn.primitive.name) >> pp_params(params_without_jaxpr) >>
         pp(' '.join(names[x] if isinstance(x, Var) else str(x.val)
                     for x in eqn.inputs)))
  return vcat([lhs >> pp(' = ') >> rhs,
               pp_jaxpr(eqn.params['jaxpr']).indent(2)])
pp_rules[xla_call_p] = pprint_xla_call

