## `Data` base class

This submodule implements the `Data` base class, which provides unified functionality and interfaces for all classes in `cmlkit` that use and store non-trivial amounts of data, for instance computed representations or kernel matrices. In the future, it will also be used for the `Dataset` class.

`Data` implements two important pieces of functionality: 

1. I/o: It knows how to save itself to disk using `.dump(filename)`, and `Data` instances can be loaded using `cmlkit.load_data(filename)`. 
2. History: `Data` instances can track which `Components` are applied to them in order, using a hash of the component. Since `Components` act like pure functions, an initial hash and the history uniquely identify a given `Data` instance. Therefore, a hash of the history can be used instead of a costly hash of the `Data` itself. This is used extensively in the caching framework.

In the parlance of `Data`, the `protocol` is an integer identifying the method to use for storing data on disk, in an attempt to ensure some flexibility for the future. At the moment, the supported protocols are:
- `0`: `.npz`, uncompressed
- `1`: `.npz`, compressed

You can pass this as keyword argument to `dump`. Loading will automatically detect the protocol to use.

Currently, `Data` exists somewhat awkwardly alongside the `engine.inout` module, and the `Dataset` class. The roadmap for the future is to convert `Dataset` into a proper `Data` subclass. It might also be useful to combine `load_data`, `from_yaml` and `read_npy` into a `cmlkit.load` uni-loader.

### Developer information

If you write a custom `Component` that is not a subclass of `Representation`, you have to assume that incoming data will be a `Data` instance, and you have to make sure your output is one as well. Typically, this means that together with your `Component` you will also have to write a simple subclass of `Data` that describes the kind of data you are returning. See `representations/data.py` or `regression/data.py` for examples.

When you create your result `Data` instance, you should use the `result` class method of `Data`, which will take care of tracking the computation history.

The current unwritten rule is to have your `Data` subclasses implemented in a `data.py` file so it rests in its own little namespace, and to give subclasses descriptive, but not overly verbose names.
