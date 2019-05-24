"""Loss functions.

A loss function (`lossf`) in `cmlkit` is a function that
takes in ground truth values (`true`), predicted values (`pred`)
and optionally an uncertainty value (`pv`) and produces a single output.

This output is typically a single number, but in some special situations
it can also be an array of the same `len` as `pred`/`true`.

Since we typically want to compute many `lossf` at once (think of the classic
`rmse`, `mae`, `r2` triplet of losses), we also introduce the concept of a more
general `Loss`, which returns a dict of `{"lossname": value, ...}`.

On the user-facing side, the typical interface to compute losses is provided
by `get_loss`. In situations where only a single loss function is needed (for instance
optimisation, the `get_lossf` interface should be used instead.) You can rely on
a `loss` always returning a dict, and a `lossf` always either a float or an array.

"""

from .lossfs import get_lossf
from .loss import get_loss
