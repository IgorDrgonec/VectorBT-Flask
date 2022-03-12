#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../docs" || exit

pip uninstall -y pdoc_to_md
pip install -U git+https://github.com/polakowo/pdoc-to-md.git
pip install -U git+https://github.com/squidfunk/mkdocs-material-insiders.git
python generate_api.py
mkdocs serve
rm -rf docs/api/