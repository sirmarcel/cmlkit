class Cached(type):
    """Meta-class for Components that support caching.

    Based on this approach:
    https://stackoverflow.com/questions/16017397/injecting-function-call-after-init-with-decorator

    This class makes sure to run cache registration *after*
    the initalisation process of a Component is complete.

    The reason this is needed is that registering a cache
    requires `get_config()` to work, and that is only
    guaranteed after all initialisation has happened. In
    particular in sub-classed Components like Representations,
    the __init__ of the parent class gets called *before* all
    the setup is done in the child class!
    """

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.register_cache()
        return obj
