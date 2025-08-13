#!/bin/bash

cd "$(dirname "$0")" # change to the directory where the script is located

find . -type f -name '*.html' -delete

rm -rf _images _modules _static _sources .doctrees

rm objects.inv searchindex.js .buildinfo

echo "Build directory cleaned."
