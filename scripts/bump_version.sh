#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit

V=${1:?"Missing version"}
perl -i -pe "s/.*/__version__ = \"$V\"/ if \$.==3" ../vectorbtpro/_version.py
perl -i -pe "s/.*/              $V/ if \$.==37" ../docs/overrides/partials/header.html
curl "https://img.shields.io/badge/version-$V-ff69b4" -o ../assets/badges/version.svg
