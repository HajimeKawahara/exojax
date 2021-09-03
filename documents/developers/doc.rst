Generate Documents
==============================

We use the google style of the sphinx document.

.. code:: sh
	  
    pip install sphinx_rtd_theme sphinxcontrib.napoleon sphinxemoji

This is an example to generate the sphinx doc.

.. code:: python3

    python setup.py install
    rm -rf documents/exojax
    sphinx-apidoc -F -o documents/exojax src/exojax
    cd documents
    make clean
    make html
