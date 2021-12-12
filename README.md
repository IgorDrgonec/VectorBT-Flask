<div align="center">
	<br>
	<img src="static/header.svg">
	<br>
</div>
<br>
<p align="center">
    <a href="https://github.com/polakowo/vectorbt.pro/blob/main/setup.py" alt="Python Versions">
        <img src="https://github.com/polakowo/vectorbt.pro/blob/main/static/badges/python_version.svg" /></a>
    <a href="https://github.com/polakowo/vectorbt/blob/master/LICENSE.md" alt="License">
        <img src="https://github.com/polakowo/vectorbt.pro/blob/main/static/badges/license.svg" /></a>
    <a href="https://discord.gg/63jZZzCwzp" alt="Discord">
        <img src="https://img.shields.io/discord/918629562441695344?logo=discord" /></a>
</p>

vectorbt PRO is the most powerful toolkit for backtesting, algorithmic trading, and research. It is a high-performance superset of the **[vectorbt](https://github.com/polakowo/vectorbt)** library, one of the world's most innovative open-source backtesting frameworks. vectorbt PRO extends the standard vectorbt library with new impressive features and useful enhancements.

## [Home](https://vectorbt.pro) · [Features](#features) · [License](#license) · [Installation](#install) · [Support](#support)

## Features

- [x] **Parallelization with Numba**: Most Numba-compiled functions have been refactored to process columns in parallel using [Explicit Parallel Loops](https://numba.pydata.org/numba-doc/0.37.0/user/parallel.html#explicit-parallel-loops), which can be enabled by simply passing `parallel=True`.
- [x] **Chunking**: Innovative chunking mechanism that takes a specification of how arguments should be chunked, automatically splits them, passes each chunk to the function, and merges back the results. This way, you can run any function in a distributed manner! Additionally, vectorbt PRO provides the chunking specification for all arguments of most Numba-compiled functions. Chunking can be enabled by simply passing `chunked=True`. No more out-of-memory errors!
- [x] **Multithreading**: Integration of the Dask backend for running multiple chunks across multiple threads. Cuts down execution time on Apple M1 by 3-4x, and even more, depending on the number of cores. Dask + Numba = :muscle:
- [x] **Multiprocessing**: Integration of the Ray backend for running multiple chunks across multiple processes. Ever wanted to test billions of hyperparameter combinations in a matter of minutes? This is now possible by scaling the simulation up in the cloud.
- [x] **Smart broadcasting**: Broadcasting mechanism has been completely refactored and now supports parameters - almost every function can build a product of multiple hyperparameter combinations with a single line of code.
- [x] **Config**: Config classes have been optimized. The time overhead of using the top-level API has shrunk by 25%.

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

vectorbt PRO is a totally different beast compared to vectorbt - a vanilla version available publicly. In fact, the PRO version redesigns the underlying core to enable groundbreaking features. 

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
