---
title: Installation
---

# Installation

!!! info
    vectorbt PRO is a totally different beast compared to the community version. 
    In fact, the PRO version redesigns the underlying core to enable groundbreaking features. 
    
    To avoid using an outdated code, make sure to only import **vectorbtpro**!

## Requirements

After you've been added to the list of collaborators and accepted the
repository invitation, the next step is to create a [Personal Access Token] for
your GitHub account in order to access the PRO repository programmatically 
(from the command line or GitHub Actions workflows):

1. Go to https://github.com/settings/tokens
2. Click on [Generate a new token]
3. Enter a name (such as "vectorbtpro")
4. Set the expiration to a fixed number of days
5. Select the [`repo`][scopes] scope
6. Generate the token and save it in a safe place

    [Personal Access Token]: https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token
    [Generate a new token]: https://github.com/settings/tokens/new
    [scopes]: https://docs.github.com/en/developers/apps/scopes-for-oauth-apps#available-scopes

### TA-Lib

To use TA-Lib for Python, you need to have the [TA-Lib](https://github.com/mrjbq7/ta-lib#dependencies) 
already installed.

To install the TA-Lib in Google Colab, run the following:

```plaintext
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
!pip install Ta-Lib
```

## With pip

The PRO version can be installed with `pip`.

Uninstall the community version if installed:

```sh
pip uninstall vectorbt
```

Install the base PRO version (with recommended dependencies):

```sh
# if you're using Git/HTTPS authentication
pip install -U "vectorbtpro[base] @ git+https://github.com/polakowo/vectorbt.pro.git"

# if you are connecting to GitHub with SSH
pip install -U "vectorbtpro[base] @ git+ssh://github.com/polakowo/vectorbt.pro.git"
```

!!! info
    Whenever you are prompted for a password, paste the token that you generated in the previous steps.

    To avoid re-entering the token over and over again, you can 
    [add it to your system](https://stackoverflow.com/a/68781050) 
    or set an environment variable `GH_TOKEN` and then install the package as follows:

    ```sh
    pip install -U "vectorbtpro[base] @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
    ```

Lightweight version (with only required dependencies):

```sh
pip install -U git+https://github.com/polakowo/vectorbt.pro.git
```

For other optional dependencies, see [extra-requirements.txt](https://github.com/polakowo/vectorbt.pro/blob/main/extra-requirements.txt).

### Google Colab

Set your token using `%env`:

```plaintext
%env GH_TOKEN=abcdef...
```

!!! warning
    Make sure to delete this cell when sharing the notebook with others!

Install vectorbt PRO:

```plaintext
!pip install -U "vectorbtpro[base] @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
```

Restart the runtime, and you're all set!

### As Python dependency

With [setuptools](https://setuptools.readthedocs.io/en/latest/) adding vectorbt PRO as a 
dependency to your Python package can be done by listing it in setup.py or in your 
[requirements files](https://pip.pypa.io/en/latest/user_guide/#requirements-files):

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

## With git

Of course, you can pull vectorbt PRO directly from `git`:

```sh
git clone git@github.com:polakowo/vectorbt.pro.git vectorbtpro
```

Install the package:

```sh
pip install -e vectorbtpro
```

## Troubleshooting

* [TA-Lib](https://github.com/mrjbq7/ta-lib#dependencies)
* [Jupyter Notebook and JupyterLab](https://plotly.com/python/getting-started/#jupyter-notebook-support)
* [Apple M1](https://github.com/polakowo/vectorbt/issues/320)