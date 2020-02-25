# this is not a place of honour


def convert_sequence(s):
    """Convert arguments to qmmlpack format"""

    if isinstance(s, (str, type(None), float, int)):
        return s
    else:
        if len(s) == 1:
            return s[0]
        elif len(s) == 2:
            if isinstance(s[1], (tuple, list)):
                # guards against accidentally passing an already
                # properly formed sequence
                return (s[0], s[1])
            else:
                return (s[0], (s[1],))
        else:
            return (s[0], s[1:])


def tuples_to_lists(d):
    """In nested object, convert all tuples to lists, in place."""

    # Get an iterator so we can go through both keys and values.
    # (Unfortunately, list and dict have different interfaces!)
    if isinstance(d, (tuple, list)):
        it = enumerate(d)
    elif isinstance(d, dict):
        it = d.items()

    # recursively convert
    for k, v in it:
        if isinstance(v, (dict, list)):
            tuples_to_lists(v)

        elif isinstance(v, tuple):
            d[k] = list(v)
            tuples_to_lists(d[k])


def lists_to_tuples(d):
    """In nested object, convert all lists to tuples, in place."""

    # Get an iterator so we can go through both keys and values.
    # (Unfortunately, list and dict have different interfaces!)
    if isinstance(d, (tuple, list)):
        it = enumerate(d)
    elif isinstance(d, dict):
        it = d.items()

    # recursively convert
    for k, v in it:
        if isinstance(v, (dict, tuple)):
            lists_to_tuples(v)

        elif isinstance(v, list):
            # the order is important here,
            # tuples cannot be changed, so
            # we need to convert from the
            # 'bottom' of the recursion
            lists_to_tuples(d[k])
            d[k] = tuple(v)


def find_key_apply_f(d, key, f):
    """Find key in dict, apply f to the value

    Example:
        d = {'arg1': 3,
             'arg2': 2}

        find_key_apply_f(d, 'arg2', lambda x: x**2) results in

         d = {'arg1': 3,
              'arg2': 4}

    Note that this replacement happens inline.

    Args:
        d: Dict to process
        key: Key to match
        f: Function to apply

    Returns:
        None; replacement is done in-place

    """

    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                d[k] = f(v)

            if isinstance(v, dict):
                find_key_apply_f(v, key, f)

            if isinstance(v, (list, tuple)):
                for i in v:
                    find_key_apply_f(i, key, f)


def find_pattern_apply_f(d, pattern, f):
    """Find values in a dict matching a given pattern, apply f to them

    Pattern in this context is a function returning True or False.

    Args:
        d: Dict to process
        pattern: Boolean function, if true, f is applied to the WHOLE key
        f: Function

    Returns:
        None; replacement is done in-place

    """

    if isinstance(d, dict):
        for k, v in d.items():
            if pattern(v) is True:
                d[k] = f(v)

            elif isinstance(v, dict):
                find_pattern_apply_f(v, pattern, f)

            elif isinstance(v, (list, tuple)):
                for i in range(len(v)):
                    if pattern(v[i]) is True:
                        v[i] = f(v[i])
                    else:
                        find_pattern_apply_f(v[i], pattern, f)

    if isinstance(d, (list, tuple)):
        for i in range(len(d)):
            if pattern(d[i]) is True:
                d[i] = f(d[i])
            else:
                find_pattern_apply_f(d[i], pattern, f)



def find_pattern(d, pattern, ignore=[]):
    """In a nested dict/list data structure, find all paths where pattern returns true.

    In a data structure composed of dicts, lists and tuples, evaluate each value,
    and if pattern returns true, record the sequence of keys leading to this point.

    Args:
        d: Dict/list/tuple data structure
        pattern: Boolean function
        ignore: For any of the paths in ignore, stop search a branch when
                the current path matches it.

    Returns:
        A list of paths, where a path is a list of keys that, when applied in sequence,
        bring you to a node where pattern returns true.

    """

    results = []

    _find_pattern(d, pattern, results, ignore=ignore)

    return results


def _find_pattern(d, pattern, results, path=[], ignore=[]):
    # recursive part; should not have to be called standalone
    if matches_any(path, ignore):
        return None
    else:
        if pattern(d) is True:
            results.append(path)

        else:
            if isinstance(d, dict):
                keys = d.keys()
            elif isinstance(d, (list, tuple)):
                keys = list(range(len(d)))
            else:
                keys = None

            if keys is not None:
                for k in keys:
                    _find_pattern(d[k], pattern, results, path + [k], ignore=ignore)


def matches_any(path, paths):
    """If path matches any path in paths, return true."""

    for p in paths:
        if p == path:
            return True

    else:
        return False


def set_with_path(d, path, value):
    """In a nested dict/list data structure, set the entry at path.

    In a data structure composed of dicts, lists and tuples, apply each
    key in sequence, then set the resulting entry of the tree to value.
    This is done in-place!

    Args:
        d: Dict/list/tuple data structure
        path: List of keys
        value: Value to set

    """

    my_d = d

    for key in path[:-1]:
        my_d = my_d[key]

    my_d[path[-1]] = value


def get_with_path(d, path):
    """In a nested dict/list data structure, set the entry at path.

    In a data structure composed of dicts, lists and tuples, apply each
    key in sequence, return the result.

    Args:
        d: Dict/list/tuple data structure
        path: List of keys

    Returns:
        Whatever is in the data structure at that path.

    """

    my_d = d

    for key in path:
        my_d = my_d[key]

    return my_d
