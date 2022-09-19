.PHONY: build-all build-ubuntu2204 run-ubuntu2204

build-all: build-ubuntu2204

UBUNTU2204_DOCKERFILE   = onednn-ubuntu2204.dockerfile
UBUNTU2204_IMAGENAME    = onednn
UBUNTU2204_IMAGEVERSION = ubuntu-22.04

build-ubuntu2204:
	$(eval DOCKERFILE := $(UBUNTU2204_DOCKERFILE))
	$(eval NAME := $(UBUNTU2204_IMAGENAME))
	$(eval VERSION := $(UBUNTU2204_IMAGEVERSION))
	docker build --tag $(NAME):$(VERSION) \
	             --force-rm \
	             --file $(DOCKERFILE) \
	             .

run-ubuntu2204:
	$(eval IMAGE := $(UBUNTU2204_IMAGENAME):$(UBUNTU2204_IMAGEVERSION))
	$(eval RUNTIMENAME := onednn-ubuntu2204)
	$(eval PWD := $(shell pwd))
	docker run --name $(RUNTIMENAME)     \
	           --hostname $(RUNTIMENAME) \
	           --rm -it                  \
	           -v $(PWD):/mnt:ro         \
	           $(IMAGE)
