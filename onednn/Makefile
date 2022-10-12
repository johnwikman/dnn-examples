.PHONY: all clean

BUILD-DIR=/tmp/dnnbuild

SOURCE-DIRS=$(shell find . -mindepth 1 -type d -not -name '_*' -exec basename \{} \;)

CXX = g++
CXXFLAGS = -ldnnl -Wall --std=c++17 -O2 -I../common/include

all: $(foreach d, $(SOURCE-DIRS), $(d).compile)

%.compile: FORCE
	$(eval SRC-FILES := $(shell find $* ../common/src/any ../common/src/onednn -type f -name "*.cpp"))
	$(eval OUTFILE := $(BUILD-DIR)/$*)
	mkdir -p $(BUILD-DIR)
	$(CXX) $(SRC-FILES) $(CXXFLAGS) -o $(OUTFILE)

%/: FORCE
	make $*.compile

FORCE:

clean:
	rm -rf $(BUILD-DIR)
