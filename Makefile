PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
WEED_PRESENT := $(wildcard weed/.)

ifeq ("$(wildcard /usr/local/bin/cmake)", "/usr/local/bin/cmake")
CMAKE_L := /usr/local/bin/cmake
else
ifeq ("$(wildcard /usr/bin/cmake)", "/usr/bin/cmake")
CMAKE_L := /usr/bin/cmake
else
CMAKE_L := cmake
endif
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build-deps         to build Weed-Loader C++ dependencies"
	@echo "  install            to install Weed-Loader"
	@echo "  wheel              to build the Weed-Loader wheel"
	@echo "  dist               to package the source distribution"

.PHONY: build-deps
build-deps:
	rm -rf weed_loader/weed_system/weed_lib
	rm -rf weed_loader/weed_system/weed_cl_precompile
ifneq ($(OS),Windows_NT)
ifeq ($(WEED_PRESENT),)
	git clone https://github.com/vm6502q/weed.git
endif
	mkdir -p weed/build
ifeq ($(UNAME_S),Linux)
	cd weed/build; $(CMAKE_L) ..; make weed_shared weed_cl_precompile
endif
ifeq ($(UNAME_S),Darwin)
	cd weed/build; cmake -DENABLE_OPENCL=OFF ..; make weed_shared weed_cl_precompile
endif
endif
	mkdir weed_loader/weed_system/weed_lib; cp weed/build/libweed_shared.* weed_loader/weed_system/weed_lib/; cd ../../..
	mkdir weed_loader/weed_system/weed_cl_precompile; cp weed/build/weed_cl_precompile weed_loader/weed_system/weed_cl_precompile/; cd ../../..

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install Weed-Loader you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist
