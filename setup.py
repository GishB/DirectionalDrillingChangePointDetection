from setuptools import setup, find_packages

__version__ = "0.0.2"

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dd_cpd',
    version=__version__,
    python_requires='>=3.10.0',
    url='https://github.com/GishB/DirectionalDrillingChangePointDetection',
    license='GNU GPLv3',
    author='Aleksandr Samofalov',
    author_email='SamofalovWORK@yandex.ru',
    description='Time Series Change Point Detection for Directional Drilling Optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=[
        'tests*',
        'experiments*',
        'docs*',
        '.*',
        '*.egg-info',
        'build*',
        'dist*'
    ]),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Operating System :: OS Ubuntu 2022 TLS',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/GishB/DirectionalDrillingChangePointDetection/issues',
        'Source': 'https://github.com/GishB/DirectionalDrillingChangePointDetection',
    },
    keywords=[
        'oil and gas',
        'change point detection',
        'directional drilling',
        'time series analysis'
    ],
)