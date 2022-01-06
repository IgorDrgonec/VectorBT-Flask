#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/../docs" || exit
python generate_api.py
mkdocs gh-deploy --force --config-file mkdocs.insiders.yml
rm -rf docs/api/
rm -rf site/