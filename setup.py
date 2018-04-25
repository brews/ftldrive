import os
from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = '1a1'
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)
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
                    install_requires=['numpy', 'pandas', 'numba'],
                    packages=find_packages(),
                    )

setup(**setup_kwargs)
