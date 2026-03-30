PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
QRACK_PRESENT := $(wildcard qrack/.)
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
ifneq ($(OS),Windows_NT)
ifeq ($(QRACK_PRESENT),)
	git clone https://github.com/unitaryfund/qrack.git; cd qrack; git checkout 164b33eedde500eb54057075094a1349e124c708; cd ..
endif
	mkdir -p qrack/build
ifeq ($(UNAME_S),Linux)
ifneq ($(filter $(UNAME_P),x86_64 i386),)
	cd qrack/build; $(CMAKE_L) -DCPP_STD=14 -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DQBCAPPOW=8 ..; make qrack qrack_cl_precompile; cd ../..
else
	cd qrack/build; $(CMAKE_L) -DCPP_STD=14 -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DENABLE_COMPLEX_X2=OFF -DENABLE_SSE3=OFF -DQBCAPPOW=8 ..; make qrack qrack_cl_precompile; cd ../..
endif
	mkdir -p qrack/build/qrack
	cp -r qrack/include/* qrack/build/qrack
	cp -r qrack/build/include/* qrack/build/qrack
endif
ifeq ($(UNAME_S),Darwin)
ifneq ($(filter $(UNAME_P),x86_64 i386),)
	cd qrack/build; $(CMAKE_L) -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCPP_STD=14 -DENABLE_OPENCL=OFF -DQBCAPPOW=8 ..; sudo make install; cd ../..
else
	cd qrack/build; $(CMAKE_L) -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCPP_STD=14 -DENABLE_OPENCL=OFF -DENABLE_RDRAND=OFF -DENABLE_COMPLEX_X2=OFF -DENABLE_SSE3=OFF -DQBCAPPOW=8 ..; sudo make install; cd ../..
endif
endif
	rm -rf weed_loader/weed_system/weed_lib
	rm -rf weed_loader/weed_system/weed_cl_precompile
ifeq ($(WEED_PRESENT),)
	git clone https://github.com/vm6502q/weed.git; cd weed; git checkout 37097ca457915d383c2768a59e1c25758571e90f; cd ..
endif
	mkdir -p weed/build
ifeq ($(UNAME_S),Linux)
	cd weed/build; $(CMAKE_L) -DWEED_TCAPPOW=6 -DWEED_CPP_STD=14 -DQRACK_INCLUDE="../qrack/build" -DQRACK_DIR="../qrack/build" ..; make weed_shared weed_cl_precompile; cd ../..
endif
ifeq ($(UNAME_S),Darwin)
ifneq ($(filter $(UNAME_P),x86_64 i386),)
	cd weed/build; cmake -DWEED_ENABLE_OPENCL=OFF -DWEED_TCAPPOW=6 -DWEED_CPP_STD=14 -DOpenBLAS_INCLUDE_DIRS="/opt/homebrew/opt/openblas/include" ..; make weed_shared weed_cl_precompile; cd ../..
else
	cd weed/build; cmake -DWEED_ENABLE_OPENCL=OFF -DWEED_TCAPPOW=6 -DWEED_CPP_STD=14 -DBLAS_LIBRARIES="-framework Accelerate" ..; make weed_shared weed_cl_precompile; cd ../..
endif
endif
	mkdir weed_loader/weed_system/weed_lib; cp weed/build/libweed_shared.* weed_loader/weed_system/weed_lib/
	mkdir weed_loader/weed_system/cl_precompile; cp weed/build/weed_cl_precompile weed_loader/weed_system/cl_precompile/; cp qrack/build/qrack_cl_precompile weed_loader/weed_system/cl_precompile/
endif

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
