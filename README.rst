diffsky
============

Installation
------------
To install diffsky into your environment from the source code::

    $ cd /path/to/root/diffsky
    $ python setup.py install

## Documentation

Online documentation for diffsky is available at 
[diffsky.readthedocs.io](https://diffsky.readthedocs.io/en/latest/).

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffsky
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

