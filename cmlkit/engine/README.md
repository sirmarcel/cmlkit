## `cmlkit` backend

This is the beating heart of `cmlkit`. Welcome! ⚙️

#### Contents

- `config.py`, `component.py`, `configparse.py`: Infrastructure for `Components`
- `data/`: Data class (used to pass results between `Components`)
- `cache/`: Heavy-duty caching infrastructure
- `hashing.py`: Computes hashes via `joblib`
- `inout.py`: i/o module (somewhat deprecated with the addition of `Data`)
- `caching.py`: Caching wrappers (somewhat deprecated with the more recent `cache` infrastructure)

If you're having a `cmlkit` read-through, I'd recommend reading in roughly that order!

### `Components`

The core object in `cmlkit` is the `Component`, describing objects pretending they're functions. A `Component`, once instantiated with all of its parameters, doesn't change and deterministically returns the same output when given the same input.

`Components` can be serialised into and instantiated from a dictionary formatted in a special way, which we call "`config` dictionaries". Their syntax is specified in `confiparse.py`, but essentially the format is:

``` 
{"class_name": { ... # arguments to __init__ }}
```

This config dictionary, since `Components` are not expected to have state, fully determines a `Component`. This aspect of a `Component` is implemented via the `Configurable` mixin class. De-serialisation from `config` to `Component` is achieved with the `cmlkit.from_config` method, which performs the class lookup.

Sometimes, it is necessary to specify some arguments that do not influence the outcome of the computation, but determines how it's executed (for instance how many cores to use, or cache settings). For this purpose, `Components` can be passed an optional `context` dictionary.

So when implementing custom components, the rules are:

- The config must fully determine the output,
- Everything else goes in the context.

Why am I doing things this way? Here are some reasons:

- This gives a consistent way to share model architectures as plaintext (yaml)
- This makes hashing and caching a breeze (no trying to hash a complex object)
- It gives full control over serialisation (I don't understand pickling at all)
- It makes a plugin system trivial to write

`config.py` implements the "de/serialise to dictionary" part of this. (`configparse.py` does the parsing.)
`component.py` adds the `context` bit, and provides the general base class for `cmlkit` components.

### Custom Components

Here are the rules for implementing your own `Components`:

- All context variables you use must have a default value set in the `default_context` of the class, to ensure that passing an empty context will never result in an error.
- Sub-classes must pass the context into the parent `__init__`.
- As explained above, `Components` must be deterministic, i.e. for the same config and the same input, the same output must result. There should be no internal state!
- If you want to use caching, your `Component` must use `Data` objects as input and output, and you should follow the instructions given in `cache/`!
- You must ensure that `_get_config(self)` returns a dictionary that fully recovers your `Component` when passed to the `_from_config` class method. (`_from_config` defaults to `__init__(**config)`).
- Make sure the config only contains types that are pleasant to dump to `yaml`, and is relatively small. (i.e. don't drop huge amounts of data into a `config`)

You're also encouraged to make the `Component` callable, i.e. using the `__call__` method, but that's up to you.

### Passing data around

`Components` typically act on instances of the `Data` class, which are essentially collections of `numpy` arrays. `Data` implements saving/loading of data, and tracks the components applied to a given set of data, for caching purposes. (See `data` submodule for details.)

### Caching

The `caching.py` in here implements some run-of-the-mill function wrappers, which are nice for one-off code, but should not be used in any serious setting, as they will eventually run into edge cases.

For serious caching of `Component` outputs, see the `cache` submodule. It makes full use of the `Component` concept, and avoids wasteful computation of hashes!
