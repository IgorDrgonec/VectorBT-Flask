#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../docs" || exit

pip uninstall -y pdoc_to_md
pip install -U git+https://github.com/polakowo/pdoc-to-md.git
python generate_api.py
mkdocs gh-deploy --force
rm -rf docs/api/
rm -rf site/