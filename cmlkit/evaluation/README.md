## Evaluation 🏫

The main interest here is in the `loss` module, which provides loss functions and infrastructure for "generalised loss functions".

`evaluator` implements the general interface for a class of `Components` called `Evaluators`, which take a `config` as input and produce some sort of diagnostic output as a `dict`. `Evaluators` also play a major role in the `cmlkit.tune` module, please take a look there to see more specialised usecases!