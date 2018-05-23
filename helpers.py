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

    """

    for k, v in d.items():
        if k == key:
            d[k] = f(v)

        if isinstance(v, dict):
            find_key_apply_f(v, key, f)

        if isinstance(v, list):
            for i in v:
                find_key_apply_f(i, key, f)
