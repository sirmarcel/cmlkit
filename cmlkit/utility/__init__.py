"""Kitchen sink of various convenient things.

Intended to keep `cmlkit.engine` reasonably streamlined.

The rough rule is 'essential code that doesn't explicitly
relate to physics in `engine`, other essential code in its
own submodule, everything else that does not fit here'
"""

from .elements import charges_to_elements
from .timing import timed, time_repeat
from .conversion import convert, unconvert
from .indices import fourway_split, threeway_split, twoway_split
from .humanhash import humanize
from .opt_lgs import OptimizerLGS
