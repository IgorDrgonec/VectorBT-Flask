<div align="center">
	<br>
	<img src="static/header.svg">
	<br>
</div>
<br>
<p align="center">
    <a href="https://github.com/polakowo/vectorbt.pro/blob/main/setup.py" alt="Python Versions">
        <img src="https://github.com/polakowo/vectorbt.pro/blob/main/static/badges/python_version.svg" /></a>
    <a href="https://github.com/polakowo/vectorbt.pro/blob/master/LICENSE.md" alt="License">
        <img src="https://github.com/polakowo/vectorbt.pro/blob/main/static/badges/license.svg" /></a>
    <a href="https://discord.gg/63jZZzCwzp" alt="Discord">
        <img src="https://img.shields.io/discord/918629562441695344?logo=discord" /></a>
</p>

vectorbt PRO is the most powerful toolkit for backtesting, algorithmic trading, and research. It is a high-performance superset of the **[vectorbt](https://github.com/polakowo/vectorbt)** library, one of the world's most innovative open-source backtesting frameworks. vectorbt PRO extends the standard vectorbt library with new impressive features and useful enhancements.

## [Home](https://vectorbt.pro) · [Features](#features) · [License](#license) · [Installation](#install) · [Support](#support)

## Features

Additionally to the [features](https://github.com/polakowo/vectorbt#zap-features) offered by the vanilla version of vectorbt, vectorbt PRO implements the following enhancements:

### Performance

- [x] **Parallelization with Numba**: Most Numba-compiled functions have been rewritten to process columns in parallel using [Explicit Parallel Loops](https://numba.pydata.org/numba-doc/0.37.0/user/parallel.html#explicit-parallel-loops), which can be enabled by a single command.
- [x] **Chunking**: Innovative chunking mechanism that takes a specification of how arguments should be chunked, automatically splits them, passes each chunk to the function, and merges back the results. This way, you can run any function in a distributed manner! Additionally, vectorbt PRO provides the chunking specification for all arguments of most Numba-compiled functions. Chunking can be enabled by a single command. No more out-of-memory errors!
- [x] **Multithreading**: Integration of the Dask backend for running multiple chunks across multiple threads. Cuts down execution time on Apple M1 by 3-4x, and even more depending on the number of cores. Dask + Numba = :muscle:
- [x] **Multiprocessing**: Integration of the Ray backend for running multiple chunks across multiple processes. Ever wanted to test billions of hyperparameter combinations in a matter of minutes? This is now possible by scaling functions and entire applications up in the cloud using the Ray cluster. :eyes:
- [x] **Jitting**: Jitting means just-in-time compiling. Although Numba remains the primary jitter in vectorbt, vectorbt PRO now enables implementation of custom jitter classes such as that for [JAX](https://github.com/google/jax) with GPU support. Every jitted function is registered globally so you can switch between implementations using a single command.
- [x] **Caching**: Caching has been reimplemented from the ground up and now it's being managed by a single registry. This allows for tracking useful statistics of all cacheable parts of vectorbt, such as to display the total cached size in MB. Full control and transparency.

### Flexibility

- [x] **Smart broadcasting**: Broadcasting mechanism has been completely refactored and now supports parameters. Build a product of multiple hyperparameter combinations with a single line of code. :magic_wand:
- [x] **Meta methods**: Many methods such as rolling apply are now available in two flavors: regular (instance methods) and meta (class methods). Regular methods work as usual, while meta methods are not bound to any array and act as micro-pipelines with their own broadcasting and templating logic.

### Data

- [x] **Refactored data**: In case of connectivity issues, data won't be lost but returned so it can be updated later. Also, symbol fetching methods can also return a state, which will be preserved for the use in data updates. There is also a new progress bar for symbols.
- [x] **Local data**: Added data classes that specialize in loading data from files, such as HDF5 or CSV. 

### Modeling

- [x] **More checks**: Introduced typing and OHLC violation checks.
- [x] **Initial position**: Similar to initial cash, initial position can now be specified.
- [x] **Cash deposits**: Cash can now be deposited/withdrawn at any time. 
- [x] **Cash earnings**: Cash earnings and dividends can now be added/removed at any time.
- [x] **Shortcut properties**: Portfolio simulation based on order functions can take user-defined arrays, write them in place during the simulation, and use them directly instead of calculating them during the analysis phase. This is the new, very convinient way to pre-calculate and utilize attributes such as returns. Most attributes in Portfolio and Records have also become properties, while their respective methods have got a "get_" prefix.

### Analysis

- [x] **Hyperfast rolling metrics**: Rolling metrics based on returns have been optimized for best performance (up to 100x speedup).
- [x] **Flexible portfolio attributes**: Portfolio attributes such as asset flow now allow overriding other attributes they are based on. This allows great control of post-simulation analysis, for example, to eliminate re-calculation.
- [x] **Redesigned records**: Every records class now has a config that describes each field of the NumPy records array. This enables many useful automatisms, such as type checks and construction of human-readable DataFrames. This also makes it far easier to subclass other records classes.

### Settings

- [x] **Optimized config**: Config classes have been optimized. The time overhead of using the top-level API has shrunk by 25%.
- [x] **New settings**: Settings have become more intuitive and human-readable. You can also use environment variables to instruct vectorbt to load settings from a file before any function is registered internally. This has some interesting use cases, such as overriding Numba options for specific functions and even replacing entire implementations in the deepest layers of the vectorbt's core.

## License

### See [LICENSE.md](https://github.com/polakowo/vectorbt.pro/tree/master/LICENSE.md) for details

### Important notes

Installing vectorbt PRO requires visiting the **https://vectorbt.pro** website and obtaining a vectorbt PRO license. The license gives the access to the vectorbt PRO codebase in a private GitHub repository.

> **It is illegal to publish, distribute, or sell the vectorbt PRO source code without a separate permission. Violation of the licensing terms will trigger a ban followed by a legal pursuit.**
>
> The vectorbt PRO is hosted in a private repository on GitHub. The access to the repository is licensed and granted by invitation only on a paid basis. In order to access the repository, the users must obtain prepaid subscription plans at **https://vectorbt.pro**. The users pay for the continued access to the repository, including updates, support and maintenance (new exchanges, improvements, bugfixes and so on).
>
> If your vectorbt PRO license expires, your software or system will not break down and will keep working fine with your most recent version by that time. However, if you discontinue your paid license you will lose the updates that will follow.
>
> Any licensed user, developer, team, or company, having obtained paid access to the vectorbt PRO repository from us, can use vectorbt PRO as a dependency, subject to the terms and limitations of the vectorbt PRO paid subscription plans.
>
> Licensees can use, copy, and modify vectorbt PRO as long as they<br />**DO NOT VENDOR, PUBLISH, SELL, OR DISTRIBUTE THE SOURCE CODE OF VECTORBT PRO**.
>
> It is allowed to specify vectorbt PRO as a dependency of your software as long as you<br />**DO NOT INCLUDE A COPY OF THE VECTORBT PRO SOURCE CODE IN YOUR SOFTWARE**.
>
> If you are a software developer you should specify vectorbt PRO as your requirement. The end-user of your software is responsible for obtaining his own individual vectorbt PRO license. The best practice is to make it clear in your docs or on your website. Since vectorbt and vectorbt PRO are interchangeable, auto-detection can be factored-in to let the end-user choose between the free vectorbt and the paid vectorbt PRO.
>
> Thank you for using vectorbt PRO legally :heart:

## Installation

---
**NOTE**

vectorbt PRO is a totally different beast compared to the open-source version of vectorbt. In fact, the PRO version redesigns the underlying core to enable groundbreaking features. 

To avoid using an outdated code, make sure to only use **vectorbtpro**!

---

Uninstall the vanilla version (if installed):

```bash
pip uninstall vectorbt
```

Install the PRO version:

```bash
# if you're using Git/HTTPS authentication
pip install -U "vectorbtpro[full] @ git+https://github.com/polakowo/vectorbt.pro.git"

# if you are connecting to GitHub with SSH
pip install -U "vectorbtpro[full] @ git+ssh://github.com/polakowo/vectorbt.pro.git"
```

Base version (minimal version for completing most examples):

```bash
pip install -U "vectorbtpro[base] @ git+https://github.com/polakowo/vectorbt.pro.git"
```

Lightweight version (minimal version for running vectorbt):

```bash
pip install -U git+https://github.com/polakowo/vectorbt.pro.git
```

For more details, see [extra-requirements.txt](https://github.com/polakowo/vectorbt.pro/blob/main/extra-requirements.txt).

### Python dependency

With [setuptools](https://setuptools.readthedocs.io/en/latest/) adding vectorbt PRO as a dependency to your Python package can be done by listing it in setup.py or in your [requirements files](https://pip.pypa.io/en/latest/user_guide/#requirements-files):

```python
# setup.py
setup(
    # ...
    install_requires=[
        "vectorbtpro @ git+https://github.com/polakowo/vectorbt.pro.git"
    ]
    # ...
)
```

## Support

## [Wiki](https://github.com/polakowo/vectorbt.pro/wiki) · [Documentation](https://vectorbt.pro/docs) · [New issue](https://github.com/polakowo/vectorbt.pro/issues) · <sub>[![Discord](https://img.shields.io/discord/918629562441695344?logo=discord&logoColor=white)](https://discord.gg/63jZZzCwzp)</sub> · [info@vectorbt.pro](mailto:info@vectorbt.pro)

© 2021 vectorbt PRO
