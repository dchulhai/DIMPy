import setuptools
from dimpy._version import __version__

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dimpy",
    version=__version__,
    author="Dhabih V. Chulhai",
    author_email="chulhaid@uindy.edu",
    description="A Discrete Interaction Model for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license_file = ('LICENSE'),
    python_requires='>=3.6',
)
