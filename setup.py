#!/usr/bin/env python
import codecs
import os
import re
from setuptools import find_packages, setup

# PROJECT SPECIFIC
NAME = 'exojax'
PACKAGES = find_packages(where='src')
META_PATH = os.path.join('src', 'exojax', '__init__.py')
CLASSIFIERS = [
    'Programming Language :: Python',
]
INSTALL_REQUIRES = ['numpy', 'pandas', 'tqdm', 'scipy',
                    'jax>=0.2.22', 'numpyro', 'pyarrow', 'bs4', 'vaex', 'jaxopt']

# END PROJECT SPECIFIC
HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), 'rb', 'utf-8') as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string.'.format(meta=meta))


if __name__ == '__main__':
    setup(
        name=NAME,
        use_scm_version={
            'write_to': os.path.join(
                'src', 'exojax', '{0}_version.py'.format(NAME)
            ),
            'write_to_template': '__version__ = "{version}"\n',
        },
        version='1.1.2',
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
        options={'bdist_wheel': {'universal': '1'}},
    )
