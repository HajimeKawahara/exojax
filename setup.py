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
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent",
]
INSTALL_REQUIRES = [
    'numpyro',
    'jaxopt',
    'jax',
    "hitran-api",
    'git+https://github.com/radis/radis.git@develop',
    "pygments>=2.15",
    "pydantic<2.0",
]

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
        version='1.4.1',
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
        python_requires='>=3.9',
        package_dir={'': 'src'},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        zip_safe=False,
        options={'bdist_wheel': {
            'universal': '1'
        }},
    )

# VAEX UNISTALL and REINSTALL See Issue 2376 vaex https://github.com/vaexio/vaex/issues/2376    
import subprocess
import sys

def uninstall(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])

def reinstall(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

uninstall('vaex-core')
uninstall('vaex-astro')
uninstall('vaex-jupyter')
uninstall('vaex-ml')
uninstall('vaex-hdf5')
uninstall('vaex-server')
uninstall('vaex-viz')

reinstall('vaex')
