***********
Conventions
***********

This is a list of general conventions employed throughout this project. I hope that this makes things a bit easier to understand.

Nested Arguments
================

Nested arguments are arguments where one choice of argument requires additional, differing parameters. For instance, a weighting function in the MBTR might take an argument, while another does not. In such cases, the pattern similar to ``qmmlpack`` is used: A tuple of arguments is passed, with the first one specifying the overall choice, and additional arguments following. If no additional parameters exist, the first argument can (usually) also be employed on its own.

Tuples vs Lists
===============

Internally, tuples are used for arguments to, for instance, the MBTR. When possible, tuples should be preferred for arguments.

Coding Style
============

``cmlkit`` aims to the compliant with pep8 and the Google style guide for Python. Here is the ``pycodestlye`` config currently used::

    [pycodestyle]
    count = False
    ignore = E226,E303
    max-line-length = 120
    statistics = True


Paths
=====

Directory paths are not assumed to end in a trailing slash.