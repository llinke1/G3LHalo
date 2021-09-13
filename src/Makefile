# define the Compiler to use
GCC = g++-10 -std=c++20


NVCC = nvcc -arch=sm_75 -rdc=true -Xcompiler -fopenmp -O3

# define various compiler flags
#CFLAGS = -std=c++17 -fopenmp -W -Wall -Wpointer-arith -Wformat-nonliteral -Winit-self -g

# define directory that should include executables
BIN = ../bin/

# define directories containing header files other than /usr/include
INCLUDES = -I ./

# define library paths in addition to /usr/lib
LFLAGS= -L

# define any libraries
LIBS = -lm -lgsl -lgslcblas

# define the source files
SRCS = $(wildcard *.cpp)

# define the cuda source files
SRCS_CU = $(wildcard *.cu)

# define the executables (ADD ALL C++ EXECUTABLES HERE!!!!)
EXECS = doPrecomputations.x

# define the CUDA executables
EXECS_CU = calculateNNMap_twoPop.x testGPU.x fit_twoPop.x calculateDerivativeNNMap_twoPop.x calculateChiSquared.x calculateNNMap.x

# define the object files
OBJS = $(SRCS:.cpp=.o)

# define the cuda object files
OBJS_CU = $(SRCS_CU:.cu=.obj)

# define the dependencies, created in first compilation
# These files tell the compiler on which header files a file depends on
DEPS = $(OBJS:%.o=%.d)


# First make target, will be compiled, if "make" is called without arguments
cpu_only: $(EXECS)
	mv *.x ../bin/
	@echo Successfully compiled CPU-part of project

all:	$(EXECS) $(EXECS_CU)
	mv *.x ../bin/
	@echo Successfully compiled whole project

# Linking of all .o Files to create the EXECS
$(EXECS): $(OBJS)
	$(GCC) $(CFLAGS) $(INCLUDES) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.o) -o $@ $(LFLAGS) $(LIBS)

$(EXECS_CU): $(OBJS_CU) $(OBJS)
	$(NVCC) $(INCLUDES) $(filter-out $(EXECS_CU:.x=.obj),  $(OBJS_CU)) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.obj) -o $@ $(LFLAGS) $(LIBS)

# Include all .d files
-include $(DEPS)
#-include $(DEPS_CU)

# Build target for every single object file
# -MMD flag creates a .d file with the same name as the .o file
%.o: %.cpp
#	$(GCC) $(CFLAGS) -MMD -c $< -o $@ $(LIBS)
	$(GCC) $(CFLAGS) -c $< -o $@ $(LIBS)

%.obj: %.cu
	$(NVCC) -c $< -o $@ $(LIBS)

# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d *~ *.obj '#*'
