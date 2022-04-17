## Documentation

This project is documented using Sphinx, as described [here](https://www.sphinx-doc.org/en/master/index.html).

## Quickstart

To build the docs as HTML, first navigate to the docs folder:
```
cd docs
```
on Linux, you can then run the following command:
```
make html 
```
alternatively, or on Windows, you can run:
```
sphinx-build . _build/html
```
The built HTML docs will then be available in the `_build/html` folder, and can be opened with any browser.

## Development

Sphinx requires docs and docstrings to be written in RST format. A good starting point on this can be found 
[here](https://sphinx-tutorial.readthedocs.io/step-1/).

### Auto-building the docs

When working on documentation, to avoid having to rebuild the docs each time a change is made, `sphinx-autobuild` can 
be used. This can be installed using:
```
pip install sphinx-autobuild
```
and used by running the following command:
```
sphinx-autobuild . _build/html
```