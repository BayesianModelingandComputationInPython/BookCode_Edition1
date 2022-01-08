#/bin/sh
pip3 install -r requirements-docs.txt &&
rm _build/html
jb build .
