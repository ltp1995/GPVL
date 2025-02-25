# This file is part of visvalingamwyatt.
# https://github.com/fitnr/visvalingamwyatt

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, 2021, fitnr <contact@fakeisthenewreal.org>

.PHONY: install test publish build clean cov

install:
	python setup.py install

cov:
	-coverage run --branch --source visvalingamwyatt -m unittest
	coverage report

test:
	python -m unittest
	
publish: build
	twine upload dist/*

build: | clean
	python -m build

clean:; rm -rf dist build
