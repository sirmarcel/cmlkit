## `cmlkit.engine.cache`

**Experimental!!**

This submodule implements the specialised caching infrastructure employed by `Components`. At the moment, it exists in parallel with `engine.caching` which implements general function wrappers. This submodule takes a different, more structured approach.

#### Contents

- `caches.py`: Cache management
- `cache.py`: Cache base class, defines the overall API
- `disk.py`: Disk cache using the `Data.dump` functionality
- `no.py`: Dummy cache

### Architecture

We build on the assumption that `Component`s act as pure functions, which are wholly specified by their `config`: Two instances of a `Component` with the same `config`, when applied to the same input data, will produce the same output. 

The cache is restricted to `Components` that use subclasses of `Data` as input/output. This is because the `Data` class takes care of saving files to disk, and is responsible for generating unique IDs that can be used as cache keys.

Cache lookups are done in two steps: First, we look up the component via a its `config`, and then the output via the `id` of the input data.  The first step is done by the `Component` on initialisation, and the second step is done *by the component* when it is called.

From a technical standpoint, a centralised `cmlkit` module level cache manager is responsible for keeping track of all the caches. `Components` register themselves with this manager, and get their "own" cache in return, which they can query directly with respect to inputs. This ensures that multiple instance of the same (in the sense of "same config") `Component` can use the *same* cache.

This whole charade is done in order to be able to avoid awkwardness that arises with `lru_cache`-style function wrappers: They typically are required to wrap module-level functions, which then means that the configuration of the cache needs to be set at import time. This is annoying, since we want to be able to decide on caching via the `context` of `Components`, which is not available on import! Alternatively, you could wrap the function upon instantiation, but then you lose the ability to share memory caches across instances.

The reason we say `Data.id` and not *hash* is because hashing arbitrary, large data in python is pretty expensive. Therefore, the `Data` class implements an alternative approach to generating a deterministic, unique ID, that avoids this problem. (Essentially, it tracks all the `Components` applied to the data, and since `Components` are deterministic, the output will be as well.)

### Usage (User)

As a user, you mainly interact with the caches through `context` dictionaries, through the `cache` key. So if you want to make a `Component` use a disk cache, you pass `context={"cache": "disk"}`. The location of this cache is set using the `CML_CACHE` environment variable. It will default to `cml_cache` in your current working directory.

In the future, you will also be able to pass a full `config` to have more fine-grained control over the cache.

`cache` always defaults to a dummy cache. In case you need to at some point turn it off manually, pass `{"cache": "no"}`.

Please see the caveats below!

### Usage (Developer)

As long as you are writing a subclass of `Component`, `cmlkit` auto-magically does all the housework for you, and `self.cache` will always return a `Cache` object. The API to interact with a `Cache` is defined in `cache.py`. It essentially mandates using `if key in cache` for checking, `cache.get(key)` for retrieval and `cache.submit(key, value)` for submitting results. There is also `cache.get_if_cached(key)` which either returns `None` or the cached result. (See `disk.py` for the reasoning behind this.) Since there is a dummy cache when caching is turned off, you don't have to check for it, you can always use this API.

You are **strongly** encouraged to make use of the caching facility.

You are expected to use the `id` attribute of the input `Data` instances as key. If you end up computing a result, it must be returned as a `Data` instance, using the `Data.result` class method. You should pass the component instance and the input data instance into this function, which will take care of returning a result data object with the history properly tracked.

## Caveats

- Disk cache performs NO storage monitoring or cleanup. DO NOT USE IT FOR LARGE-SCALE HYPER-PARAMETER OPTIMISATION! This WILL end badly.††
- Currently, no infrastructure exists for only caching results that take some minimum time to compute.
- In-memory caching is not implemented yet.
- It is unclear how threadsafe all of this is.

† If you try to implement this via function wrappers only, you'd be forced to assign the caches to functions defined at module level to be able to share the caches between instances of objects. This then creates problems because you lose the ability to configure the type of cache per instance, since the cache is already instantiated when the module is imported!

†† I once wrote 345TB to a scratch directory in a week or so via caching. It did not end well for anyone.
