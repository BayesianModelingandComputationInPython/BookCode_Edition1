#/bin/sh
pip3 install -r requirements-docs.txt &&
rm /opt/build/repo/_build/html -rf
jb build .
