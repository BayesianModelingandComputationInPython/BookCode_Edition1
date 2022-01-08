#/bin/sh
pip3 install -r requirements-docs.txt &&
jb clean . &&
jb build .
