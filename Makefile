all:
	g++ segmenter.cpp -I /usr/include/boost -std=c++0x -o segmenterTest

clean:
	rm *o segmenterTest
