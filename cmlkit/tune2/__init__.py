"""Hyperparameter tuning infrastructure.

Interfaces

Search:
    suggest() -> tid, suggestion.
        tid must be unique. it identifies this suggestion.
        suggestion must be a dict and should be a valid config.
    submit(tid, error=False, loss=None, var=None).
        tid must match a previously suggested one.
        if the trial failed, error must be true, loss and var are ignored.
        if the trial succeeded, error must be false, loss is required and var is optional.

Run:
    This interface is consumed by the Executor/must be followed by it.
    get_evaluation() -> eid, config.
        eid is the hash of the config and identifies this evaluation.
        config is something that the Evalutor can deal with.
    report_evaluation(result).
        result is a config-style dict. its "kind" field is either "ok" or "error".
        If it's "ok", the inner config should be:
            (see Evaluator interface)
            additional keys are ignored.
        Otherwise the inner config should be a dict:
            optional key: "error" (the class name of raised error)
            optional key: "traceback" (the traceback)
            additional keys are ignored.

Evaluator:
    __call__(config) -> result.
        result must contain the keys "loss" and "duration".
        it can contain the keys "var" and "refined_config".
        additional keys are ignored.
        exceptions must be raised, not caught.

"""
