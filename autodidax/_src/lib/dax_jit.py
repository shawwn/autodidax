from argparse import Namespace as NS
from threading import local
import sys
from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class DaxGlobalJitState:
  disable_jit: Optional[bool] = False
  enable_x64: Optional[bool] = False
  extra_jit_context: Optional[Any] = None
  post_hook: Optional[Callable] = None


@dataclass
class DaxThreadLocalJitState(local):
  disable_jit: Optional[bool] = False
  enable_x64: Optional[bool] = False
  extra_jit_context: Optional[Any] = None
  post_hook: Optional[Callable] = None


def global_state():
  global global_state
  state = DaxGlobalJitState()
  def global_state_():
    nonlocal state
    return state
  global_state = global_state_
  return global_state()


def thread_local_state():
  global thread_local_state
  state = DaxThreadLocalJitState()
  def thread_local_state_():
    nonlocal state
    return state
  thread_local_state = thread_local_state_
  return thread_local_state()
