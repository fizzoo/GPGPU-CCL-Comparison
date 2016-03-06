CXXFLAGS=-Wall -Wextra -pedantic -std=c++14 -g -Og
LDLIBS=-lOpenCL -lpng

tester: Image.cc LabelData.cc Strategy.cc
