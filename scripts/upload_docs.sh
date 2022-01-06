#!/bin/bash

python ../docs/generate_api.py
mkdocs gh-deploy --force --config-file mkdocs.insiders.yml