CXXFLAGS=-Wall -Wextra -pedantic -std=c++14
LDLIBS=-lOpenCL -lpng
SRC=tester.cc Image.cc LabelData.cc Strategy.cc RGBAConversions.cc utilityCL.cc

tester: $(SRC)
	$(CXX) $(CXXFLAGS) -g $(SRC) $(LDLIBS) -o $@

fasts:  $(SRC)
	$(CXX) -DNDEBUG $(CXXFLAGS) -O3 $(SRC) $(LDLIBS) -o $@

format:
	zsh -c 'for f in *.cc *.h kernel.cl; do clang-format -i $$f; done'

clean:
	rm -f tester fasts
	rm -rf out
