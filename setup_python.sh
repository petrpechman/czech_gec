#!/usr/bin/env bash
set -xe

source ~/.bashrc
VERSION=$1

apt-get update
apt-get install -y --no-install-recommends "$VERSION" "$VERSION-venv" "$VERSION-distutils" "$VERSION-dev"

ln -sf /usr/bin/$VERSION /usr/bin/python3
ln -sf /usr/bin/$VERSION /usr/bin/python


if [[ ! -f "/usr/local/include/$VERSION" ]]; then
  ln -sf /usr/include/$VERSION /usr/local/include/$VERSION
fi

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
python3 -m pip install --no-cache-dir --upgrade pip