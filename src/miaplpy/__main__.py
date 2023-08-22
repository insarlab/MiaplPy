"""Main module to provide command line interface to the workflows."""
import sys

from .miaplpyApp import main

# https://docs.python.org/3/library/__main__.html#packaging-considerations
# allows `python -m miaplpyApp` to work
sys.exit(main())
