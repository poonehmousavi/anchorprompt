"""ExpeL knowledge pool for LALM robustness.

Pipeline:  data -> attacks -> lalm (actor) -> reflect (POOL UPDATE) -> pool -> evaluate
The pool is written in exactly one place: `src.expel.reflect.update_pool_from_failures`.
"""
