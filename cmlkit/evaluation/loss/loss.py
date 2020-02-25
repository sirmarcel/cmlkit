"""The Loss.

This module and class aim to provide a somewhat useable interface
for the problem of specifying multiple loss functions to compute.

In principle, the canonical way is to provide a list of strings with
names of loss functions. But in practice, we also would like the ability
to use pre-defined collections of lossfs as shortcuts. And sometimes, we
only want to have a single loss, so a list doesn't make sense.

`get_loss` provides an interface to use all of these. The resulting
`Loss` instance then acts as the wrapper to the individual loss functions.
Its arguments are found in a `spec`, from which it can be re-created.

Note that I have declined to use the `config` machinery of `cmlkit` here,
since I think it would be overkill to do so -- the lossfs don't have any
parameters (at least currently). So the `spec` it is, which is just a list
of strings, the same that the constructor would expect.

***

As a user, you should know that typically, when some method in `cmlkit` asks
you for a `loss` argument, you can supply anything that `get_loss` can handle.

If you are instead asked for `lossf`, it needs to be one of the functions in
the `lossf` module.

"""

from .lossfs import get_lossf, lossfs

shortcuts = {"default": ["rmse", "mae", "r2"], "all": list(lossfs.keys())}


class Loss:
    """Loss wraps multiple loss functions.

    Attributes:
        needs_pv: If True, at least one of the wrapped lossfs needs
            an uncertainty measure.
        longname: Long name of this loss.
        spec: Arguments that can re-create this loss.

    """

    def __init__(self, *args):
        self.lossfs = [get_lossf(l) for l in args]
        self.needs_pv = any(l.needs_pv for l in self.lossfs)
        self.longname = str(self.spec)

    @property
    def spec(self):
        return [l.__name__ for l in self.lossfs]

    def __call__(self, true, pred, pv=None):
        """Compute multiple losses and returns them in dict."""

        return {l.__name__: l(true, pred, pv=pv) for l in self.lossfs}


def get_loss(*args):
    """Obtain a loss.

    Multiple formats are supported:
        get_loss("rmse", "mae") -> Loss("rmse", "mae")
        get_loss(["rmse", "mae"]) -> Loss("rmse", "mae")
        get_loss("default")  -> Loss("rmse", "mae", "r2")
        get_loss(Loss("rmse")) -> Loss("rmse")

    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, Loss):
            return arg
        elif isinstance(arg, (list, tuple)):
            args = arg
        elif arg in shortcuts:
            args = shortcuts[arg]

    return Loss(*args)
