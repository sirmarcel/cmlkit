## How to run optimisations

Before you start, you need two things: A `Search` instance, and an `Evaluator`. If you don't quite know what that's supposed to mean, have a brief skim of the `README.md` file!

For a more interactive and detailed tutorial, please have a look at the `cmlkit` tutorial mentioned in the main readme.

### Search

The `search` module will tell you about this in detail, but in essence, you need to define a search space using `hyperopt` terminology and then pass that to the `Hyperopt` class. 

For example:

```
space = {"x": ["hp_uniform", "x", 0, 1]}
search = Hyperopt(space)
```

This means that you'd like to sample `x` uniformly from the reals between 0 and 1. `search.suggest()` will now return examples from that distribution:

```
search.suggest()                                                                            
>>> (0, {'x': 0.7280300912545745})

search.suggest()                                                                            
>>> (1, {'x': 0.8586937697592628})

search.suggest()                                                                            
>>> (2, {'x': 0.9925181073994515})

search.suggest()                                                                           
>>> (3, {'x': 0.6734772542773243})

search.suggest()                                                                           
>>> (4, {'x': 0.5626149179372755})
```

In reality, the space should be the "skeleton" of an entire model. 

For example:

```python
{
  "model": {
    "per": "mol",
    "regression": {
      "krr": {
        "kernel": {
          "kernel_atomic": {
            "norm": False,
            "kernelf": {
              "gaussian": {"ls": ["hp_loggrid", "ls_start", -13, 13, 1.0]}
            },
          }
        },
        "nl": ["hp_loggrid", "nl_start", -18, 0, 1.0],
      }
    },
    "representation": {
      "soap": {
        "elems": [1, 6, 7, 8, 9],
        "n_max": ["hp_choice", "n_max", [1, 2, 3, 4, 5, 6, 7, 8]],
        "l_max": ["hp_choice", "l_max", [1, 2, 3, 4, 5, 6, 7, 8]],
        "cutoff": ["hp_choice", "cutoff", [2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "sigma": ["hp_loggrid", "sigma", -20, 6, 53],
      }
    },
  }
}
```

### Evaluator

You'll also need an evaluator. This can be something very simple:

```python

class MockEvaluator(Component):
    kind = "quadratic_eval"

    def __call__(self, model):
        loss = (model["x"] - 0.5) ** 2

        return {"loss": loss}

    def _get_config(self):
        return {}

```

In reality, you can maybe use one of the already implemented `Evaluators`, for instance `TuneEvaluatorHoldout`, which trains a `Model` on a training set and then computes the loss on a holdout test set.

## Instantiating a `Run`

One final ingredient is that you need a `config` defining your stopping strategy. For now, the only existing one is:

```python
search = {"stop_max": {"count": 100}}
```

I.e. simply counting the number of trials. 

Now we're ready:

```python
run = Run(search, evaluator, stop)
```

Then, we need to prepare the run:

```python
run.prepare()
```

Which will generate a folder with a random, unique run name. Then, we can simply run for, let's say, 5 minutes:

```python
run.run(duration=5)
```

Once finished, the run write out its current top results into the folder, and terminate. While it's running, you can see the current state in the `status.txt` file in the run folder. You can also inspect `tape.son` to see how the run is stored under the hood!

And that's basically it. You can restore a run from a folder by calling `Run.restore(rundir)`, or get a read-only version with `Run.checkout(rundir)`. 

The number of workers can be set through the context at instantiation, so you'll need to call:

```python
run = Run(search, evaluator, stop, context={"max_workers": 40})
```

