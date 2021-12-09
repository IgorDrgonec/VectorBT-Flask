from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8", errors='ignore') as fh:
    long_description = fh.read()

version = {}
with open("vectorbt/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)

setup(
    name='vectorbt',
    version=version['__version__'],
    description='Supercharged backtesting and technical analysis for quants',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/polakowo/vectorbt',
    packages=find_packages(),
    package_data={
        'vectorbt': ['templates/*.json']
    },
    install_requires=[
        'numpy>=1.16.5',
        'pandas',
        'numba==0.53.1; python_version == "3.7"',
        'numba>=0.53.1; python_version != "3.7"',
        'scipy',
        'scikit-learn',
        'schedule',
        'requests',
        'tqdm',
        'dateparser',
        'imageio',
        'pytz',
        'typing_extensions; python_version < "3.8"',
        'mypy_extensions',
        'humanize',
        'attrs'
    ],
    extras_require={
        'data': [
            'yfinance>=0.1.63',
            'python-binance',
            'ccxt',
            'tables'
        ],
        'ta': [
            'ta',
            'pandas_ta',
            'TA-Lib',
        ],
        'acc': [
            'Bottleneck',
            'numexpr',
        ],
        'exec': [
            'ray>=1.4.1',
            'dask'
        ],
        'plot': [
            'matplotlib',
            'plotly>=4.12.0',
            'ipywidgets>=7.0.0'
        ],
        'stats': [
            'quantstats>=0.0.37',
            'PyPortfolioOpt'
        ],
        'misc': [
            'python-telegram-bot>=13.4',
            'dill'
        ],
        'cov': [
            'pytest',
            'pytest-cov',
            'codecov'
        ],
        'docs': [
            'pdoc3'
        ]
    },
    python_requires='>=3.6, <3.10',
    license='Apache 2.0 with Commons Clause',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: Free for non-commercial use',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
)
