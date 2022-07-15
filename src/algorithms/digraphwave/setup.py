#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

PACKAGE_NAME = "digraphwave"


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


# from pathlib import Path
# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()


setup(
    name=PACKAGE_NAME,
    version='0.0.0',
    description='Calculate matrix exponential for Laplacian matrices.',
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    author='Ciwan Ceylan',
    author_email='ciwan@kth.se',
    url=f'https://github.com/ciwanceylan/{PACKAGE_NAME}',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        # 'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
        'Private :: Do Not Upload',
    ],
    project_urls={
        'Documentation': f'https://{PACKAGE_NAME}.readthedocs.io/',
        'Changelog': f'https://{PACKAGE_NAME}.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': f'https://github.com/ciwanceylan/{PACKAGE_NAME}/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.15', 'scipy>=1.0.0', 'numba>=0.50.1', 'torch>=1.11.0', 'torch-sparse>=0.6.13',
    ],
    extras_require={
        # 'numba': ['numba>=0.50.1']
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    }
)
