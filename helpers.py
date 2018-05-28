def convert_sequence(s):
    """Convert arguments to qmmlpack format"""

    if isinstance(s, str):
        return s
    else:
        if len(s) == 1:
            return s[0]
        elif len(s) == 2:
            return (s[0], (s[1],))
        else:
            return (s[0], s[1:])


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
