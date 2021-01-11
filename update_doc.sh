python setup.py install
rm -rf documents/exojax
sphinx-apidoc -F -o documents/exojax src/exojax
cd documents
make clean
make html
