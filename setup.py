#!/usr/bin/env python3
import sys

# Python supported version checks
if sys.version_info[:2] < (3, 10):
    raise RuntimeError('Python version >= 3.10 required.')

import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='hungry_moose',
    version='1.0.0',  # Required
    description='We\'re gonna be rich basically',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    author='Roflez Mahn, Jessie Purser',
    author_email='mahnriley@gmail.com, jessiepurser@gmail.com',
    keywords='bitch, tits, dick',
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'': 'hungry_moose'},
    packages=find_packages(where='hungry_moose'),
    python_requires='>=3.10',
    # TODO: The following would provide a command called `main_titty` which
    # executes the function `main_titty` from hungry_moose when invoked:
    entry_points={
        'console_scripts': [
            'main_titty=hungry_moose:main_titty',
            'train_titty=hungry_moose:train_titty',
        ],
    },
    project_urls={
        'slack': 'https://piffsquad.slack.com/',
        'jira': 'https://piffsquad.atlassian.net/',
        'confluence!': 'fillinlater',
        'source': 'https://github.com/Roflz/stock-prediction',
    },
)



