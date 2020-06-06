# http://mrbook.org/blog/tutorials/make/
# http://peter.bourgon.org/blog/2009/10/10/gnu-make-and-deleting-intermediate-files.html
# https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html
# http://stackoverflow.com/questions/18668003/the-procedure-entry-point-gxx-personality-v0-could-not-be-located-in-the-dnyam
# compile only with the c option
# g++ -c optlib.cc cum_normal.cc cum_normal_bivariate.cc N3.cc normdist.cc

# link object files into an executable
# g++ -shared -Wl,-soname,optlib.so -o optlib.so optlib.o cum_normal.o cum_normal_bivariate.o N3.o normdist.o

CC=g++
CFLAGS=-c -fPIC

SOURCES=src/optlib.cc src/stats.cc
#OBJECT=obj/$(SOURCES:.c=.o)
OBJECT=obj/optlib.o obj/stats.o
INCLUDE=-Iinc
EXECUTABLE=../../zipline/finance/valuation/optlib

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECT)
	#mac
	$(CC) -shared -Wl,-install_name,$(EXECUTABLE).so -o $(EXECUTABLE).so $(OBJECT)
	#linux
	#$(CC) -shared -Wl,-soname,$(EXECUTABLE).so -o $(EXECUTABLE).so $(OBJECT)

obj/%.o: src/%.cc
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $<
	
