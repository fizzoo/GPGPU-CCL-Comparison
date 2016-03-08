CXXFLAGS=-Wall -Wextra -pedantic -std=c++14
LDLIBS=-lOpenCL -lpng
SRC=tester.cc Image.cc LabelData.cc Strategy.cc

tester: $(SRC)
	g++ $(CXXFLAGS) -g -Og $(SRC) $(LDLIBS) -o $@

fasts:  $(SRC)
	g++ -DNDEBUG $(CXXFLAGS) -O3 $(SRC) $(LDLIBS) -o $@

format:
	zsh -c 'for f in *.cc *.h; do clang-format -i $$f; done'

clean:
	rm -f tester fasts
	rm -rf out
