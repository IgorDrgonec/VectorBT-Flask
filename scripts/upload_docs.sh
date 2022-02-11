#!/bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately
cd "$(dirname "${BASH_SOURCE[0]}")/../docs" || exit

echo "Updating pdoc_to_md..."
pip uninstall -y pdoc_to_md
pip install -U git+https://github.com/polakowo/pdoc-to-md.git

echo "Generating API..."
python generate_api.py

echo "Building static files..."
mkdocs build --clean

echo "Locking pages..."
python lock_pages.py

echo "Pushing static files to GitHub..."
python mkdocs_cli.py gh-deploy --force

echo "Locking notebooks..."
python lock_notebooks.py

echo "Pushing locked content to GitHub..."
git add ../locked-pages.md
git add ../locked-notebooks.md
git commit -m "Update locked content"
git push

echo "Cleaning up..."
rm -rf docs/api/
rm -rf site/