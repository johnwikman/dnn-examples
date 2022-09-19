.PHONY: all

BUILD-DIR=/tmp/build

SOURCE-DIRS=$(shell find . -mindepth 1 -type d -exec basename \{} \;)

CXX = g++
CXXFLAGS = -ldnnl -Wall -O2

all:
	echo "TODO"
	echo $(SOURCE-DIRS)

%.compile:
	$(eval SRC-FILES := $(shell ls $*/*.cpp))
	$(eval OUTFILE := $(BUILD-DIR)/$*)
	mkdir -p $(BUILD-DIR)
	$(CXX) $(SRC-FILES) $(CXXFLAGS) -o $(OUTFILE)

clean:
	rm -rf $(BUILD-DIR)
