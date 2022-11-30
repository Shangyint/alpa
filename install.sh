#!/usr/bin/bash

pip3 install -e ".[dev]"

cd build_jaxlib
python3 build/build.py --enable_cuda --dev_install --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa
cd dist

pip3 install -e .