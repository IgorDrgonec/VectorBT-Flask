#!/bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately
cd "$(dirname "${BASH_SOURCE[0]}")/../docs" || exit

C='\033[1;32m'
NC='\033[0m' # No Color

echo "${C}Updating pdoc_to_md...${NC}"
pip uninstall -y pdoc_to_md
pip install -U git+https://github.com/polakowo/pdoc-to-md.git

echo "${C}Generating API...${NC}"
python generate_api.py

echo "${C}Building static files...${NC}"
mkdocs build --clean

echo "${C}Locking pages...${NC}"
python lock_pages.py

echo "${C}Pushing static files to GitHub...${NC}"
python mkdocs_cli.py gh-deploy --force

echo "${C}Cleaning up...${NC}"
rm -rf docs/api/
rm -rf site/