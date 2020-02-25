## `cmlkit.tune`

This is the hyperparameter tuning component of `cmlkit`, implementing an extensible, parallel, asynchronous optimisation framework. At the moment, `hyperopt` is providing the optimisation algorithms, but in principle, any non-sequential optimiser can easily be added.

**If you just want to know how to use this, take a look at `TUTORIAL.md`!**
(But reading all of this first will help.)

Important features are:
- Robust parallelisation (on a single computer) with timeouts (i.e. we can
  deal with unstable external components that don't terminate). NO MONGODB
- Resuming/replay: Runs are saved as a series of steps, so they can always be recovered, replayed and continued. The steps are saved as the optimisation proceeds, and writes are atomic, so very little can go wrong.
- Caching of evaluations: no redundant evaluations are undertaken.
- Flexible architecture: Computation of losses, the types of models evaluated, and the stopping criteria can be customised.
- Async architecture: Recognising that model evaluation cannot be expected to take the same time, everything happens asynchronously.
- Transparent, configurable error treatment: Errors can be optionally caught and reported.
- Separation of concerns: As opposed to `hyperopt`, details of the execution are not visible to the optimisation algorithm, the executor doesn't need to know how to compute losses, and so on.
- Use `hyperopt` without having to touch its internals 😉


### Architecture

We separate the task of optimising into multiple components:

1. A `Search`, which makes suggestions based on previous results, i.e. the part that you'd colloquially call "optimisation", (see `search` for the `hyperopt` wrapper)
2. An `Evaluator` which computes losses (i.e. stores the data needed to compute losses, takes care to train the models, etc.), and (see `evaluators` for examples)
3. A `Run` instance that ties everything together and executes the optimisation (see `run`).

These components communicate with each other in narrowly-defined ways, and avoid conceptual crossover between these distincts concerns. As a result, this architecture is quite general, and easy to extend in well-defined ways.

For users and developers, the most likely extension will be providing custom `Evaluators` to model specific training/prediction procedures, or special losses. Please see below for an overview of the interface you must implement. Also note that it is crucial that `Evaluators` are `Components` -- the communication with parallel workers also happens via the `config` mechanism, so `cmlkit` must know how to instantiate your `Evaluator`!

From a technical standpoint, most work is done in the `run` submodule, as the whole goal of this architecture is to minimise the involvement of the `Search` and `Evaluator` in the actual running of the optimisation. If you're interested in the technical details, please have a look there. It's neat! 😎

### Interfaces

If you implement custom `Evaluators`/`Searches`, here is the interface
that they need to implement.

```
Search:
    suggest() -> tid, suggestion.
        tid must be unique. it identifies this suggestion.
        suggestion must be a dict and should be a valid config.
    submit(tid, error=False, loss=None, var=None).
        tid must match a previously suggested one.
        if the trial failed, error must be true, loss and var are ignored.
        if the trial succeeded, error must be false, loss is required and var is optional.

    Searches must be deterministic with suggest and submit are performed in the same order with the same arguments.

Evaluator (must be a Component):
    __call__(config) -> result.
        result must contain the keys "loss" and "duration".
        it can contain the keys "var" and "refined_config".
        additional keys are ignored.
        exceptions must be raised, not caught.

```

### Caveats

- We assume that evaluations aren't so expensive they need checkpointing, and aren't much faster than about 1s. The latter can be customised to some extent, but there is a builtin assumptions that the event loop of a `Run` doesn't run too fast.
- Currently, re-running from `son` is slow because we use the `yaml` dumper. It is slightly more readable, but at least 10x slower than `json`. This will be changed.

### Roadmap

- `MPI` support. In principle it is easy to extend this architecture to multi-node parallelisation. Preliminary work is already done, but it's currently not a priority.
- `json` instead of `yaml`. 

### Why `tape`? Why all of this? Why not just use `hyperopt`?

Basically, I couldn't find out how to cleanly store the internal state of a `hyperopt` optimisation to do restarts. If you use the `mongoDB` backend, you can simply use the database, but due to the constraints of the systems I have to work on, running a database server is not really feasible. (It's also way too complicated for such a "simple" task.) But if you simply rerun `submit` and `suggest` in the same order, you can recover the state, and you don't have to touch the internals of `hyperopt`.

Apart from that, now you have very fine-grained control over the initialisation of the workers. So for instance, you can load datasets exactly once, upon startup of the worker, and not once per evaluation. The `hyperopt` native worker implementations was a bit difficult to come to terms with, and was very insistent on always shutting down the workers after each evaluations, which is very wasteful.

You also don't have to write the function to minimise by hand, as you'd have to do for `hyperopt`, which is error prone, and it's easy to run into awkward situations with pickling and parallelisation.
