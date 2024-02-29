# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='truncnormkde',
    version='0.1.0',
    description='''Implements boundary-unbiased kde's in JAX that do not impose a zero derivative at the boundary''',
    long_description=readme,
    author='Asad Hussain',
    author_email='asadh@utexas.edu',
    url='https://github.com/potatoasad/truncnormkde',
    packages=find_packages(exclude=('tests', 'docs', 'dev'))
)