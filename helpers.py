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
