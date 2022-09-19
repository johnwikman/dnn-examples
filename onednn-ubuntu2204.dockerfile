FROM ubuntu:22.04

# Install dependencies and clear cache
RUN export DEBIAN_FRONTEND=noninteractive             \
 && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
 && apt update -y                                     \
 && apt upgrade -y                                    \
 && apt install -y cmake g++ git nano


# Build & Install oneDNN (setting -j2 here to avoid crash...)
RUN export CC=gcc CXX=g++                              \
 && cd /tmp                                            \
 && git clone https://github.com/oneapi-src/oneDNN.git \
 && cd oneDNN                                          \
 && git checkout v2.6.2                                \
 && mkdir -p build                                     \
 && cd build                                           \
 && cmake ..                                           \
 && make -j2                                           \
 && ctest                                              \
 && cmake --build .                                    \
 && cmake --install . --prefix /usr                    \
 && cd /tmp                                            \
 && rm -rf oneDNN

WORKDIR /root

CMD ["/bin/bash"]
