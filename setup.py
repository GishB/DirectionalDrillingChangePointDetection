__version__ = "0.0.1"

from setuptools import setup, find_packages

setup(
    name='dd_cpd',
    version=__version__,
    python_requires='>=3.10.0',
    url='https://github.com/GishB/DirectionalDrillingChangePointDetection',
    license='GNU GPLv3',
    packages=find_packages(exclude=[
        'tests',
        'experiments',
        '.github',
        '.git',
        '__pycache__',
        '.pytest_cache',
        '.idea',
        '.git',
        'gitattributes'
                                    ]),
    author='Aleksandr Samofalov',
    author_email='SamofalovWORK@yandex.ru',
    description='Time Series Change Point Detection '
                'to increase perfomance of Directional Drilling Processes at Oil and Gas Fields',
    long_description=open('./README.md').read(),
    install_requires=[
        'pandas~=1.5.3',
        'pytest~=8.1.1',
        'numpy~=1.25.0',
        'streamlit<=1.31.0',
        'scipy~=1.11.4',
        'matplotlib~=3.7.1',
        'requests~=2.31.0',
        'detecta<=0.0.5'
        'tsad==0.19.3'
    ],
    include_package_data=True,
    zip_safe=False)
