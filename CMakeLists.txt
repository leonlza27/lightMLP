#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully

cmake_minimum_required(VERSION 4.0)

project(test)

ADD_SUBDIRECTORY(mlpCCore)
ADD_SUBDIRECTORY(langbind)
