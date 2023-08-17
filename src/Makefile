# define the Compiler to use
GCC = g++ -std=c++20

NVCC = nvcc -arch=sm_75 -rdc=true -Xcompiler -fopenmp -O3

# define various compiler flags
CFLAGS = -fopenmp -W -Wall -Wpointer-arith -Wformat-nonliteral -Winit-self -g

# define if GPU acceleration is used
GPU = true

ifeq ($(GPU), true)
USERFLAGS += -DGPU=true
endif

# define if Scaling of Satellite variance is used
SCALING = false
ifeq ($(SCALING), true)
USERFLAGS += -DSCALING=true
endif

# define if you want the verbose code version
VERBOSE = true

ifeq ($(VERBOSE), true)
USERFLAGS += -DVERBOSE=true
endif

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
EXECS_CU = calculateNNMap_twoPop.x testGPU.x calculateNNMap.x

# define the object files
OBJS = $(SRCS:.cpp=.o)

# define the cuda object files
OBJS_CU = $(SRCS_CU:.cu=.obj)

# define the dependencies, created in first compilation
# These files tell the compiler on which header files a file depends on
DEPS = $(OBJS:%.o=%.d)

# First make target, will be compiled, if "make" is called without arguments
ifeq ($(GPU), true)
all:	$(EXECS) $(EXECS_CU)
	mv *.x ../bin/
	@echo Successfully compiled whole project with GPU acceleration
else
all:	$(EXECS) $(EXECS_CU)
	mv *.x ../bin/
	@echo Successfully compiled whole project without GPU acceleration
endif

# Linking of all .o Files to create the EXECS
$(EXECS): $(OBJS)
	$(GCC) $(CFLAGS) $(INCLUDES) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.o) -o $@ $(LFLAGS) $(LIBS)

ifeq ($(GPU), true)
$(EXECS_CU): $(OBJS_CU) $(OBJS)
	$(NVCC) $(INCLUDES) $(filter-out $(EXECS_CU:.x=.obj),  $(OBJS_CU)) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.obj) -o $@ $(LFLAGS) $(LIBS)
else
$(EXECS_CU): $(OBJS_CU) $(OBJS)
	$(GCC) $(CFLAGS) $(INCLUDES) $(filter-out $(EXECS_CU:.x=.obj),  $(OBJS_CU)) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.obj) -o $@ $(LFLAGS) $(LIBS)
endif


# Include all .d files
-include $(DEPS)

# Build target for every single object file
# -MMD flag creates a .d file with the same name as the .o file
%.o: %.cpp
#	$(GCC) $(CFLAGS) -MMD -c $< -o $@ $(LIBS)
	$(GCC) $(CFLAGS) $(USERFLAGS) -c $< -o $@ $(LIBS)

ifeq ($(GPU), true)
%.obj: %.cu
	$(NVCC) $(USERFLAGS) -c $< -o $@ $(LIBS)
else
%.obj: %.cu
	$(GCC) $(CFLAGS) $(USERFLAGS) -xc++ -c $< -o $@ $(LIBS)
endif

# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d *~ *.obj '#*'
