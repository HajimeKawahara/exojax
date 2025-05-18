Documentation
==============================

The ``documents`` directory is important in itself (see 
`ExoJAX docs <https://secondearths.sakura.ne.jp/exojax/>`_
), but it is also essential for building an LLM-generated wiki, such as 
`deep wiki <https://deepwiki.com/HajimeKawahara/exojax>`_
. Please make sure to update the documentation whenever you add new features.


How to generate the documents
---------------------------------

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

Structure of the documents
-----------------------------

The following shows the directory structure related to the documents.

::

  exojax/
  ├── examples/
  └── documents/
      ├── tutorials/
      └── userguide/


The ``examples`` directory directly stores the Python files to be displayed with 
`Gallery <https://secondearths.sakura.ne.jp/exojax/examples/index.html>`_
.


The ``tutorials`` and ``userguide`` directories contain reStructuredText (``rst``) files. 
While there is no major distinction between ``tutorials`` and ``userguide``, 
if you generate rst files from Jupyter Notebooks (``ipynb``) as described later, 
please store both the ipynb and the corresponding rst files in the ``tutorials`` directory.

Converts the jupyter notebooks to rst
-----------------------------------------

When creating documentation in a Jupyter Notebook, you can convert it to reStructuredText (rst) within the ``documents/tutorials`` 
directory using the following Python code:

.. code:: sh

    python jupyter2rst_each.py (ipynb filename)



Generates the up-to-date documents of tutorials
------------------------------------------------

The following commands automatically run the tutorial notebooks and generate rst:

``documents/tutorials/``


.. code:: sh

    python jupyter2rst.py exe

If you just want to generate rst without executing notebooks, try this:

.. code:: sh

    python jupyter2rst.py none

One by One:

.. code:: sh

    python jupyter2rst_each.py (python filename)

