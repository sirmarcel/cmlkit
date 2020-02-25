"""Wrapper for dealing with troublesome external function calls.

Since `cmlkit` interfaces with various external codes, we occasionally
run into a situation where an external funciton needs to be run with
a fixed timeout, and needs to be amenable to KeyboardInterrups.

Caveats:
- Only pickleable functions and arguments can be used.
- This is not particularly CPU/Mem efficient, since arguments are serialised.
- In Python < 3.8, there is a hardcoded size limit of 2GB on what can be
  passed through a Pipe to a worker process. (see https://bugs.python.org/issue17560)
  This means that this wrapper will explode spectacularly when faced with such objects.

In the future(TM) this should be rewritten to avoid pickling entirely through mem mapping.

In high-performance/throughput contexts, it might be more useful to turn off
this wrapper entirely and implement subprocess isolation at a different layer of abstraciton.

(For instance, in `cmlkit.tune`, the runner should take care of this, since then we only
need to return back losses, and need to only pass configs into the subprocess.)

"""

import os
import signal
from multiprocessing import Process, Pipe
from .errors import CmlTimeout


def wrap_external(f, timeout=None, kill=True, disable=False):
    """Wrap a function such that it is run in an external process.

    This is mainly required for external C code that might not respond to SIGTERM,
    but rather needs to be brutally SIGKILL'ed. This will run the function on a
    worker process, and kill it if a KeyboardInterrupt is detected or a timeout occurs.

    This will also only work with functions that can be pickled
    (i.e. defined at module level) and that return something that can be pickled.
    """

    if disable:
        return f

    def wrapped(*args, **kwargs):
        def f_on_worker(connection):
            # compute f on worker, send result back through Pipe
            # if an error is raised, capture it and raise it in
            # the main process.

            try:
                res = f(*args, **kwargs)
                action = "return"
            except Exception as e:
                res = e
                action = "raise"

            connection.send((action, res))

        a, b = Pipe()
        p = Process(target=f_on_worker, args=(a,))
        p.start()

        try:
            if b.poll(timeout=timeout):  # returns False if timed out
                action, res = b.recv()
                p.join(1)
                p.terminate()
                if kill and p.is_alive():
                    # this is probably overkill
                    p.join(1)
                    os.kill(p.pid, signal.SIGKILL)

                if action == "return":
                    return res
                else:
                    raise res

            else:
                p.terminate()
                p.join(1)
                if kill and p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)

                raise CmlTimeout(
                    f"Function '{f.__name__}' timed out after {timeout} seconds."
                )

        except KeyboardInterrupt:
            p.terminate()
            p.join(1)
            if kill and p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

            raise KeyboardInterrupt

    return wrapped
