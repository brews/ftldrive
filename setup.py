import os
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

MAJOR = 0
MINOR = 0
MICRO = 1
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
FULLVERSION = VERSION


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'ftldrive', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

write_version_py()

ftlcore = Extension('ftldrive.core.ekf',
                  sources=['ftldrive/core/ekf.pyx'],
                  include_dirs=[np.get_include()]
                  )

setup_kwargs = dict(name='ftldrive',
                    version=FULLVERSION,
                    description='ftldrive',
                    url='https://github.com/brews/ftldrive',
                    author='S. Brewster Malevich',
                    author_email='malevich@email.arizona.edu',
                    license='GPLv3',
                    classifiers=[
                        'Development Status :: 1 - Planning',

                        'Operating System :: POSIX',

                        'Intended Audience :: Developers',
                        'Intended Audience :: Science/Research',

                        'Topic :: Scientific/Engineering',
                        'Topic :: Software Development',

                        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

                        'Programming Language :: Python :: 3'],
                    keywords='assimilation kalman',
                    install_requires=['numpy', 'Cython', 'pandas'],
                    packages=find_packages(),
                    package_data={'ftldrive': ['tests/*.csv']},
                    )

# Deal with bad linking bug with conda in readthedocs builds.
if os.environ.get('READTHEDOCS') != 'True':
    setup_kwargs['ext_modules'] = cythonize([ftlcore])

setup(**setup_kwargs)
