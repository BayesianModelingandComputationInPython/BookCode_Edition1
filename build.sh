#/bin/sh
pip3 install -r requirements-docs.txt &&
rm -rf _build/html &&
jb build .
