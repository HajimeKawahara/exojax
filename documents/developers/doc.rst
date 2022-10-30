Generating Documents
==============================

We use the google style of the sphinx document.

.. code:: sh
	  
    pip install sphinx_rtd_theme sphinxemoji

This is an example to generate the sphinx doc.

.. code:: python3

    python setup.py install
    rm -rf documents/exojax
    sphinx-apidoc -F -o documents/exojax src/exojax
    cd documents
    make clean
    make html

Generating up-to-date documents of tutorials
------------------------------------------------

The below commands automatically run the tutorial notebooks and generate rst:

.. code:: sh

	cd documents/tutorials/
    python jupyter2rst.py exe

If you just want to generate rst without executing notebooks, try this:

.. code:: sh

    python jupyter2rst.py exe

