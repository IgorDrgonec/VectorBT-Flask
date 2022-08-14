---
title: Installation
description: How to install vectorbt PRO
---

# Installation

!!! info
    vectorbt PRO is a totally different beast compared to the open-source version. 
    In fact, the PRO version redesigns the underlying core to enable groundbreaking features. 
    
    To avoid using an outdated code, make sure to only import **vectorbtpro**!

## Requirements

### Token

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

!!! important
    After a few months you may get an email from GitHub stating that your personal access token has expired.
    If so, please go over the steps above and generate a new token.

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
!pip install TA-Lib
```

## With pip

The PRO version can be installed with `pip`.

!!! hint
    It's highly recommended creating a new virtual environment solely for vectorbtpro, such as with 
    [Anaconda](https://www.anaconda.com/).

Uninstall the open-source version if installed:

```shell
pip uninstall vectorbt
```

Install the base PRO version (with recommended dependencies) using `git+https`:

```shell
pip install -U "vectorbtpro[base] @ git+https://github.com/polakowo/vectorbt.pro.git"
```

!!! info
    Whenever you are prompted for a password, paste the token that you generated in the previous steps.

    To avoid re-entering the token over and over again, you can 
    [add it to your system](https://stackoverflow.com/a/68781050) 
    or set an environment variable `GH_TOKEN` and then install the package as follows:

    ```shell
    pip install -U "vectorbtpro[base] @ git+https://${GH_TOKEN}@github.com/polakowo/vectorbt.pro.git"
    ```

Same using `git+ssh` (see [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)):

```shell
pip install -U "vectorbtpro[base] @ git+ssh://github.com/polakowo/vectorbt.pro.git"
```

Lightweight version (with only required dependencies):

```shell
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

```shell
git clone git@github.com:polakowo/vectorbt.pro.git vectorbtpro
```

Install the package:

```shell
pip install -e vectorbtpro
```

## With Docker

Using [Docker](https://www.docker.com/) is a great way to get up and running in a few minutes, as it 
comes with all dependencies pre-installed.

[Docker image of vectorbtpro](https://github.com/polakowo/vectorbt.pro/blob/main/Dockerfile) is based on 
[Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/) -
a set of ready-to-run Docker images containing Jupyter applications and interactive computing tools.
Particularly, the image is based on [jupyter/scipy-notebook](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook),
which includes a minimally-functional JupyterLab server and preinstalled popular packages from the scientific 
Python ecosystem, and extends it with Plotly and Dash for interactive visualizations and plots,
and vectorbtpro and most of its optional dependencies. The image requires the source of vectorbtpro to be 
available in the current depository.

Before proceeding, make sure to [have Docker installed](https://docs.docker.com/install/).

Launch Docker using Docker Desktop.

### Building

Clone the vectorbtpro repository (if not already). Run this from a directory where you want
vectorbtpro to reside, for example, in Documents/GitHub:

```shell
git clone git@github.com:polakowo/vectorbt.pro.git vectorbtpro --depth=1
```

Go into the directory:

```shell
cd vectorbtpro
```

Build the image (can take some time):

```shell
docker build . -t vectorbtpro
```

Create a working directory inside the current directory:

```shell
mkdir work
```

### Running

Start a container running a JupyterLab Server on the port 8888:

```shell
docker run -it --rm -p 8888:8888 -v "$PWD/work":/home/jovyan/work vectorbtpro
```

!!! info
    The use of the `-v` flag in the command mounts the current working directory on the host 
    (`{PWD/work}` in the example command) as `/home/jovyan/work` in the container. The server logs 
    appear in the terminal. Due to the usage of [the flag --rm](https://docs.docker.com/engine/reference/run/#clean-up---rm) 
    Docker automatically cleans up the container and removes the file system when the container exits, 
    but any changes made to the `~/work` directory and its files in the container will remain intact on the host. 
    [The -it flag](https://docs.docker.com/engine/reference/commandline/run/#assign-name-and-allocate-pseudo-tty---name--it) 
    allocates pseudo-TTY.

Alternatively, if the port 8888 is already in use, use another port (here 10000):

```shell
docker run -it --rm -p 10000:8888 -v "$PWD/work":/home/jovyan/work vectorbtpro
```

Once the server has been launched, visit its address in a browser. The address is printed in
the console, for example: `http://127.0.0.1:8888/lab?token=9e85949d9901633d1de9dad7a963b43257e29fb232883908`

!!! note
    Change the port if necessary.

This will open JupyterLab where you can create a new notebook and start working with vectorbt PRO :tada:

To make use of any files on the host, put them into to the working directory `work` on the host 
and they will appear in the file browser of JupyterLab. Alternatively, you can drag and drop them 
directly into the file browser of JupyterLab.

### Stopping

To stop the container, first hit ++ctrl+c++, and then upon prompt, type `y` and hit ++enter++

### Upgrading

To upgrade the Docker image to a new version of vectorbtpro, first, update the local version 
of the repository from the remote:

```shell
git pull
```

Then, rebuild the image:

```shell
docker build . -t vectorbtpro
```

!!! info
    This won't rebuild the entire image, only the vectorbtpro installation step.

## Troubleshooting

* [TA-Lib](https://github.com/mrjbq7/ta-lib#dependencies)
* [Jupyter Notebook and JupyterLab](https://plotly.com/python/getting-started/#jupyter-notebook-support)
* [Apple M1](https://github.com/polakowo/vectorbt/issues/320)