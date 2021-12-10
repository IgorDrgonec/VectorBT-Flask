> vectorbt PRO is next-gen framework for backtesting, algorithmic trading, and research. It is a high-performance superset of the [vectorbt](https://github.com/polakowo/vectorbt) library, one of the world's most innovative open-source backtesting frameworks. vectorbt PRO includes the standard vectorbt library and wraps it with powerful new features and useful enhancements.

## [Home](https://vectorbt.pro) · [Features](#features) · [License](#license) · [Installation](#install) · [Support](#support)

## Features

## License

### See [LICENSE.md](https://github.com/polakowo/vectorbt.pro/tree/master/LICENSE.md) for details

### Important notes

> **It is illegal to publish, distribute, or sell the vectorbt PRO source code without a separate permission. Violation of the licensing terms will trigger a ban followed by a legal pursuit.**
>
> The vectorbt PRO is hosted in a private repository on GitHub. The access to the repository is licensed and granted by invitation only on a paid basis. In order to access the repository, the users must obtain prepaid subscription plans at https://vectorbt.pro. The users pay for the continued access to the repository, including updates, support and maintenance (new exchanges, improvements, bugfixes and so on).
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

> Installing vectorbt PRO requires visiting the https://vectorbt.pro website and obtaining a vectorbt PRO license. The license gives the access to the vectorbt PRO codebase in a private GitHub repository.
>
> Lightweight version:

```bash
# if you're using Git/HTTPS authentication
pip install -U git+https://github.com/polakowo/vectorbt.pro.git

# if you are connecting to GitHub with SSH
pip install -U git+ssh://git@github.com/polakowo/vectorbt.pro.git
```

> Base version:

```bash
pip install -U "git+https://github.com/polakowo/vectorbt.pro.git[base]"
```

> Full version:

```bash
pip install -U "git+https://github.com/polakowo/vectorbt.pro.git[full]"
```

For more details, see [extra-requirements.txt](https://github.com/polakowo/vectorbt.pro/blob/main/extra-requirements.txt).

### Python dependency

> With [setuptools](https://setuptools.readthedocs.io/en/latest/) adding vectorbt PRO as a dependency to your Python package can be done by listing it in setup.py or in your [requirements files](https://pip.pypa.io/en/latest/user_guide/#requirements-files):

```python
# setup.py
setup(
    # ...
    install_requires=[
        'vectorbtpro @ git+https://github.com/polakowo/vectorbt.pro.git'
    ]
    # ...
)
```

## Support

## [Wiki](https://github.com/polakowo/vectorbt.pro/wiki) · [Documentation](https://vectorbt.pro/docs) · [New issue](https://github.com/polakowo/vectorbt.pro/issues) · <sub>[![Discord](https://img.shields.io/discord/918629562441695344?logo=discord&logoColor=white)](https://discord.gg/63jZZzCwzp)</sub> · [info@vectorbt.pro](mailto:info@vectorbt.pro)

© 2021 vectorbt PRO