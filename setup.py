#!/usr/bin/env python
import codecs
import os
import re
from setuptools import find_packages, setup

# PROJECT SPECIFIC
NAME = 'ExoJAX'
PACKAGES = find_packages(where='src')
META_PATH = os.path.join('src', 'exojax', '__init__.py')
CLASSIFIERS = [
    'Programming Language :: Python',
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Operating System :: OS Independent",
]
INSTALL_REQUIRES = [
    'radis',
    'jax>=0.2.22',
    'numpyro',
    'jaxopt',
    "astropy",  # Unit aware calculations
    "astroquery>=0.3.9",  # to fetch HITRAN databases
    "beautifulsoup4",  # parse ExoMol website
    "configparser",
    "cython",
    "hitran-api",
    "lxml",  # parser used for ExoMol website
    "matplotlib",  # ">=3.4.0" to suppress the Ruler warning, but only available for Python >= 3.7
    "habanero",  # CrossRef API to retrieve data from doi
    "h5py",  # HDF5
    "hjson",
    "ipython>=7.0.0",
    "joblib",  # for parallel loading of SpecDatabase
    "json-tricks>=3.15.0",  # to deal with non jsonable formats
    "pandas>=1.0.5",
    "plotly>=2.5.1",
    "progressbar2",  # used in vaex
    "numba",
    "mpldatacursor",
    "publib>=0.3.2",  # Plotting styles for Matplotlib
    "plotly>=2.5.1",  # for line survey HTML output
    "peakutils",
    "termcolor",
    "tables",  # for pandas to HDF5 export
    "pytest",  # to run test suite
    "numba",  # just-in-time compiler
    "psutil",  # for getting user RAM
    "seaborn",  # other matplotlib themes
    "scipy>=1.4.0",
    # "tuna",  # to generate visual/interactive performance profiles
    "vaex>=4.9.2",  # load HDF5 files  (version needed to fix https://github.com/radis/radis/issues/486). #TODO : install only required sub-packages
    "lmfit",  # for new fitting modules
    "numpy<=1.22.3 ",
    "pygments>=2.15",
    "pydantic<2.0"
]

#INSTALL_REQUIRES = [
#    'numpy<=1.22.3', 'jax>=0.2.22'
#]

# END PROJECT SPECIFIC
HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), 'rb', 'utf-8') as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file,
        re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string.'.format(meta=meta))


if __name__ == '__main__':
    setup(
        name=NAME,
        use_scm_version={
            'write_to':
            os.path.join('src', 'exojax', '{0}_version.py'.format(NAME)),
            'write_to_template':
            '__version__ = "{version}"\n',
        },
        version='1.3',
        author=find_meta('author'),
        author_email=find_meta('email'),
        maintainer=find_meta('author'),
        maintainer_email=find_meta('email'),
        url=find_meta('uri'),
        license=find_meta('license'),
        description=find_meta('description'),
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=PACKAGES,
        package_dir={'': 'src'},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        zip_safe=False,
        options={'bdist_wheel': {
            'universal': '1'
        }},
    )
