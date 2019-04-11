"""Module for dealing with external bindings"""

import os
import signal
from multiprocessing import Process, Pipe
from .errors import CmlTimeout


def wrap_external(f, timeout=None, kill=True):
    """Wrap a function such that it is run in an external process, handling KeyboardInterrupts and timeouts

    This is mainly required for external C code that might not respond to SIGTERM,
    but rather needs to be brutally SIGKILL'ed. This will run the function on a
    worker process, and kill it if a KeyboardInterrupt is detected or a timeout takes place.


    Note that this *should* be not terribly memory inefficient, since most modern OSes implement
    copy-on-write (COW) which avoids copying the entire memory contents then creating a subprocess.

    This will also only work with functions that can be pickled (i.e. defined at module level),
    and that return something that can be passed through a pipe.
    """

    def wrapped(*args, **kwargs):
        def f_on_worker(connection):
            # compute f on worker, send result back through Pipe
            # if an error is raised, capture it and raise it in
            # the main process.

            try:
                res = f(*args, **kwargs)
                action = 'return'
            except Exception as e:
                res = e
                action = 'raise'

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

                if action == 'return':
                    return res
                else:
                    raise res

            else:
                p.terminate()
                p.join(1)
                if kill and p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)

                raise CmlTimeout("Function '{}' timed out after {} seconds.".format(f.__name__, timeout))

        except KeyboardInterrupt:
            p.terminate()
            p.join(1)
            if kill and p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

            raise KeyboardInterrupt

    return wrapped
