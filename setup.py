#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
#import versioneer
from os import path
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


#setup_requirements = [ ]

#test_requirements = [ ]

setup(
    author="Marcus Michael Noack",
    author_email='MarcusNoack@lbl.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python package for highly flexible function-valued Gaussian processes (fvGP)",
    entry_points={
        'console_scripts': [
            'fvgp=fvgp.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fvgp',
    name='fvgp',
    packages=find_packages(include=['fvgp', 'fvgp.*']),
    test_suite='tests',
    url='https://github.com/MarcusMichaelNoack/fvgp',
    version='2.3.1',
    zip_safe=False,
)
