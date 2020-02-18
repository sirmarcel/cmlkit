# `cmlkit` backend

This is the beating heart of `cmlkit`. Welcome! ⚙️

## The building blocks: `Component`, `config`

The main objects in `cmlkit` follow a pattern similar to Keras models:

They can be serialised into and instantiated from `config` dictionaries.
Their syntax is specified in the `config.py` module, but essentially the format is:

``` 
{"class_name": { ... # arguments to __init__ }}
```

The underlying idea is that these objects are essentially convenience wrappers
around functions with lots of arguments (for instance a regression method or
a representation), and don't have a lot of state (ideally they'd have none), so
once instantiated, they simply do some deterministic computation on inputs.

The config dict then simply specifies what kind of computation is done.

Since sometimes it is necessary to specify some arguments that do not influence
the outcome of the computation, but determines how it's executed (for instance
how many cores to use, or cache settings), `Components` also have a `context`,
another dictionary that is passed to the constructor as optional keyword argument.

So when implementing custom components, the rules are:

- The config must fully determine the output,
- Everything else goes in the context.

Why am I doing things this way? Here are some reasons:

- This gives a consistent way to share model architectures as plaintext (yaml)
- This makes hashing and caching a breeze (no trying to hash a complex object)
- It gives full control over serialisation (I don't understand pickling at all)
- It makes a plugin system trivial to write.

`config.py` implements the "de/serialise to dictionary" part of this. (`configparse.py` does the parsing.)
`component.py` adds the `context` bit, and provides the general base class for `cmlkit` components.

## Caching

The architecture above is very well-suited for caching, but it is not yet implemented. 
The main reason for this is that while `Components` are easy to hash, their inputs and outputs are not. 
This can be fixed by wrapping everything in a custom class that tracks the transformations applied to an initial dataset, 
which shouldn't be too hard, but is not done yet!

The `caching.py` in here implements some run-off-the-mill function wrappers, which are nice for one-off code, 
but should not be used in any serious setting, as they will eventually run into edge cases.
