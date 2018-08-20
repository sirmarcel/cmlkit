import os.path

__all__ = [
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__commit__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None


__title__ = "cmlkit"
__summary__ = "Tools for working with qmmlpack"
__uri__ = "https://github.com/sirmarcel/cmlkit"

__version__ = "1.0.0-beta"
__short_version__ = "1.0"

if base_dir is not None and os.path.exists(os.path.join(base_dir, ".commit")):
    with open(os.path.join(base_dir, ".commit")) as fp:
        __commit__ = fp.read().strip()
else:
    __commit__ = None

__author__ = "Marcel Langer"
__email__ = "mail@sirmarcel.com"

__license__ = "MIT License"
__copyright__ = "2018 %s" % __author__
