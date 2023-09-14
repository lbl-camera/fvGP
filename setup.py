#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
import versioneer
from os import path
"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
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
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python package for highly flexible function-valued Gaussian processes (fvGP)",
    entry_points={
        'console_scripts': [
            'fvgp=fvgp.cli:main',
        ],
    },
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser', 'myst-nb', 'sphinx-panels', 'autodocs', 'jupytext']
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='fvgp',
    name='fvgp',
    packages=find_packages(include=['fvgp', 'fvgp.*']),
    test_suite='tests',
    url='https://github.com/MarcusMichaelNoack/fvgp',
    #version='3.2.7',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
