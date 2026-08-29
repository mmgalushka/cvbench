"""On-disk dataset concerns shared by the CLI, the web API, and every task.

Kept free of TensorFlow/Keras imports on purpose: the web API imports this
package to browse dataset folders without paying for a TensorFlow import, and
``tests/test_import_boundaries.py`` enforces that this stays true.
"""
