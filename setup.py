#!/usr/bin/env python

from setuptools import setup

setup(name='ScottiePippen',
    version="0.1",
    description="Determine stellar age and mass from measurements of log L and T eff.",
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    url="https://github.com/iancze/ScottiePippen",
    license="MIT",
    packages=["ScottiePippen"],
    requires=['numpy', 'scipy', 'astropy']
    )
