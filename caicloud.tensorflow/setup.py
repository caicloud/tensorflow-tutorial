# coding=utf-8
# Copyright 2017 Caicloud authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'VERSION'), encoding='utf-8') as f:
    VERSION = f.read()

REQUIRED_PACKAGES = [
    'tensorflow >= 1.0.0',
    'grpcio >= 1.0.4',
    'googleapis-common-protos >= 1.5.0',
]

setup(
    name='caicloud.tensorflow',
    version=VERSION,
    description='Caicloud TensorFlow as a Service (TaaS) dev environment',
    long_description=long_description,
    author='Caicloud Inc.',
    author_email='taas@caicloud.io',
    url='https://caicloud.io/',
    license='Apache 2.0',
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='caicloud tensorflow serving machine learing',
    packages=find_packages(exclude=['docs', 'tests']),
)
