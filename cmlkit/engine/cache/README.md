# `cmlkit.engine.cache`

## Architecture

This submodule implements the specialised caching infrastructure employed by `Components`. At the moment, it exists in parallel with `engine.caching` which implements general function wrappers. This submodule takes a different, more rigid approach.

We build on the assumption that `Component`s act as pure functions, which are wholly specified by their `config`. I.e. two instances of a `Component` with the same `config`, when applied to the same input, will produce the same output. Therefore, cache lookups can essentially be done in two steps: First, we look up the component via a its `config`, and then the output via a hash of the input.

The first step is done by the `Component` on initialisation, and the second step is done *by the component* when it is called.

From a technical standpoint, a centralised `cmlkit` module level "hyper cache" (or cache manager) is responsible for keeping track of all the caches. `Components` register themselves with this manager, and get their "own" cache in return, which they can query directly with respect to inputs. This ensures that multiple instance of the same (in the sense of "same config") `Component` can use the *same* cache.

This whole charade is done in order to be able to avoid awkwardness that arises with `lru_cache`-style function wrappers: They typically are required to wrap module-level functions, which then means that the configuration of the cache needs to be set at import time. This is annoying, since we want to be able to decide on caching via the `context` of `Components`, which is not available on import! Alternatively, you could wrap the function upon instantiation, but then you lose the ability to share memory caches across instances.

## Usage (User)

As a user, you mainly interact with the caches through `context` dictionaries, through the `cache` key. So if you want to make a `Component` use a disk cache, you pass `context={"cache": "disk"}`. The location of this cache is set using the `CML_CACHE` environment variable. It will default to `cml_cache` in your current working directory.

In the future, you will also be able to pass a full `config` to have more fine-grained control over the cache.

`cache` always defaults to `None`. In case you need to at some point turn it off manually, pass `{"cache": None}`.

Please see the caveats below!

## Usage (Developer)

After instantiation, you need to call `cmlkit.hypercache.register(self)`. This will return a `Cache` object that you can use. If caching is not turned on, you will get None. You have to deal with this. (The reason we don't just return a non-operational cache is so you can avoid having to compute hashes when they're not needed.)

The API to interact with a `Cache` is defined in `cache.py`. It essentially mandates using `if key in cache` for checking, `cache.get(key)` for retrieval and `cache.submit(key, value)` for submitting results. There is also `cache.get_if_cached(key)` which either returns `None` or the cached result. (See `disk.py` for the reasoning behind this.)

There is no automatic function caching. This forces the "if not in cache then compute" logic into the specific components, which is slightly inelegant, but it allows us to nicely decouple the cache implementations from execution logic. It also has the advantage of making the `Component` responsible for figuring out which keys to use for the cache, which makes sense as `Components` are expected to process only specific types of data, which they should understand.

It is highly recommended to make sure that computing the keys is stable across restarts of the runtime, and not too slow. If you're caching `Datasets`, this is guaranteed by using `.geom_hash` or `.hash` as needed. The `cmlkit.engine.compute_hash()` function (backed by `joblib`) is generally stable, but slow for large `numpy` arrays.

If you implement caching for a parent class, you need to implement a mechanism to register the cache at the end of running the child class `__init__`, since otherwise you probably won't have the `config` ready. Check `representation/representation.py` for an example of this!


## Caveats

- Disk cache performs NO storage monitoring or cleanup. DO NOT USE IT FOR LARGE-SCALE HYPER-PARAMETER OPTIMISATION! This WILL end badly.††
- Currently, no infrastructure exists for only caching results that take some minimum time to compute.
- In-memory caching is not implemented yet.
- It is unclear how threadsafe all of this is.

† If you try to implement this via function wrappers only, you'd be forced to assign the caches to functions defined at module level to be able to share the caches between instances of objects. This then creates problems because you lose the ability to configure the type of cache per instance, since the cache is already instantiated when the module is imported!

†† I once wrote 345TB to a scratch directory in a week or so via caching. It did not end well for anyone.