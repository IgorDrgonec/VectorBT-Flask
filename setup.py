from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8", errors='ignore') as fh:
    long_description = fh.read()

version = {}
with open("vectorbtpro/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)

setup(
    name='vectorbtpro',
    version=version['__version__'],
    description='Next-gen framework for backtesting, algorithmic trading, and research',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/polakowo/vectorbt.pro',
    packages=find_packages(),
    package_data={
        'vectorbtpro': ['templates/*.json']
    },
    python_requires='>=3.6, <3.10',
    license='LICENSE.md',
    data_files=[('', ['LICENSE.md'])],
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Software Development',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
)
