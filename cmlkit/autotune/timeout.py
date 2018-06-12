"""A wrapper to execute Hyperopt cost functions safely on a separate process, with timeout

This code is based off parts of https://github.com/hyperopt/hyperopt-sklearn,
which falls under the following license:
=======
Copyright (c) 2013, James Bergstra
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of hyperopt-sklearn nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import

from multiprocessing import Process, Pipe
import sys
import time
import os
import signal

from hyperopt import STATUS_OK, STATUS_FAIL
import numpy as np

from cmlkit import logger
from cmlkit.autotune.objective import objective

PY2 = sys.version_info[0] == 2
int_types = (int, long) if PY2 else (int,)

def is_integer(obj):
    return isinstance(obj, int_types + (np.integer,))


def is_number(obj, check_complex=False):
    types = ((float, complex, np.number) if check_complex else
             (float, np.floating))
    return is_integer(obj) or isinstance(obj, types)


def get_vals(trial):
    """Determine hyperparameter values given a ``Trial`` object"""
    # based on hyperopt/base.py:Trials:argmin
    return dict((k, v[0]) for k, v in trial['misc']['vals'].items() if v)


def timeout_objective(d):
    """An objective function with builtin timeout.

    Note that this is VERY experimental and might lead to zombie processes etc.

    The code is based on this gist https://gist.github.com/hunse/247d91d14aaa8f32b24533767353e35d,
    which in turn is based on something found in found in https://github.com/hyperopt/hyperopt-sklearn,
    which falls under the following license

    =======
    Copyright (c) 2013, James Bergstra
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of hyperopt-sklearn nor the names of its contributors
          may be used to endorse or promote products derived from this software
          without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """


    def _cost_fn(conn):
        try:
            t_start = time.time()
            rval = objective(d)
            t_done = time.time()

            if not isinstance(rval, dict):
                rval = dict(loss=rval)
            assert 'loss' in rval, "Returned dictionary must include loss"
            loss = rval['loss']
            assert is_number(loss), "Returned loss must be a number type"
            rval.setdefault('status', STATUS_OK if np.isfinite(loss)
                            else STATUS_FAIL)
            rval.setdefault('duration', t_done - t_start)
            rtype = 'return'

        except Exception as exc:
            rval = exc
            rtype = 'raise'

        # -- return the result to calling process
        conn.send((rtype, rval))

    conn1, conn2 = Pipe()
    
    th = Process(target=_cost_fn, args=(conn2,))
    
    th.start()
    if conn1.poll(d['config']['timeout']):
        fn_rval = conn1.recv()
        th.join()
        assert fn_rval[0] in ('raise', 'return')
        if fn_rval[0] == 'raise':
            raise fn_rval[1]
        else:
            return fn_rval[1]
    else:
        logger.info("TRIAL TIMED OUT")
        
        # Attempt to normally terminate the process
        th.terminate()
        th.join(2)  # waits 2s for termination...
        logger.debug("After calling terminate() is the process alive? {}.". format(th.is_alive()))

        # If the above didn't work, we now send SIGKILL until the process dies (this is not very nice)
        i = 1
        while th.is_alive() and i <= 60:
            os.kill(th.pid, signal.SIGKILL)  # terminate sends SIGTERM which seems to not work reliably
            th.join(1)  # wait 1s
            logger.debug("After {} attempts at killing it, is the process alive? {}.". format(i, th.is_alive()))
            i += 1

        if th.is_alive():
            logger.warn('Unable to forcefully kill timed out child process with pid {}.'.format(th.pid))

        return {'status': STATUS_FAIL,
                'failure': 'timeout',
                'loss': float('inf')}

