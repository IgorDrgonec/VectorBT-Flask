---
title: Installation
---

# Installation

!!! info
    vectorbt PRO is a totally different beast compared to the open-source version of vectorbt. 
    In fact, the PRO version redesigns the underlying core to enable groundbreaking features. 
    
    To avoid using an outdated code, make sure to only use **vectorbtpro**!

## Requirements

After you've been added to the list of collaborators and accepted the
repository invitation, the next step is to create a [personal access token] for
your GitHub account in order to access the PRO repository programmatically 
(from the command line or GitHub Actions workflows):

1.  Go to https://github.com/settings/tokens
2.  Click on [Generate a new token]
3.  Enter a name and select the [`repo`][scopes] scope
4.  Generate the token and store it in a safe place

  [personal access token]: https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token
  [Generate a new token]: https://github.com/settings/tokens/new
  [scopes]: https://docs.github.com/en/developers/apps/scopes-for-oauth-apps#available-scopes

## With pip

The PRO version can be installed with `pip`.

Uninstall the vanilla version if installed:

```sh
pip uninstall vectorbt
```

Install the base PRO version:

```sh
# if you're using Git/HTTPS authentication
pip install -U "vectorbtpro[base] @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"

# if you are connecting to GitHub with SSH
pip install -U "vectorbtpro[base] @ git+ssh://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
```

The `GH_TOKEN` environment variable must be set to the value of the personal access 
token you generated in the previous step. Note that the personal access token must 
be kept secret at all times, as it allows the owner to access your private repositories.

Full version (with all optional dependencies):

```sh
pip install -U "vectorbtpro[full] @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
```

Lightweight version (with only required dependencies):

```sh
pip install -U git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git
```

For more details, see [extra-requirements.txt](https://github.com/polakowo/vectorbt.pro/blob/main/extra-requirements.txt).

### Google Colab

Set your token using `%env`:

```plaintext
%env GH_TOKEN=abcdef...
```

Install vectorbt PRO in the next cell:

```plaintext
!pip install -U "vectorbtpro[full] @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
```

### As Python dependency

With [setuptools](https://setuptools.readthedocs.io/en/latest/) adding vectorbt PRO as a 
dependency to your Python package can be done by listing it in setup.py or in your 
[requirements files](https://pip.pypa.io/en/latest/user_guide/#requirements-files):

```python
# setup.py
setup(
    # ...
    install_requires=[
        "vectorbtpro @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
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

* [TA-Lib support](https://github.com/mrjbq7/ta-lib#dependencies)
* [Jupyter Notebook and JupyterLab support](https://plotly.com/python/getting-started/#jupyter-notebook-support)