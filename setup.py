#!/usr/bin/env python

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup

setup(name='ScottiePippen',
    version="0.1",
    description="Determine stellar age and mass from measurements of log L and T eff.",
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    packages=["ScottiePippen"],
    requires=['numpy', 'scipy']
    )
