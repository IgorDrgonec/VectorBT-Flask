#!/bin/bash

V=${1:?"Missing version"}
cd "$(dirname "${BASH_SOURCE[0]}")" || exit
perl -i -pe "s/.*/__version__ = \"$V\"/ if \$.==3" ../vectorbtpro/_version.py
curl "https://img.shields.io/static/v1\?label\=version\&message\=$V\&color\=ff69b4" -o ../assets/badges/version.svg
