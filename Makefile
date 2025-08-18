# Compiler settings
CXX = g++
NVCC = nvcc

# Find all source files
CC_SOURCES := $(wildcard *.cc)
CU_SOURCES := $(wildcard *.cu)
HEADERS := $(wildcard *.h)

# Convert sources to object files
CC_OBJECTS := $(CC_SOURCES:.cc=.o)
CU_OBJECTS := $(CU_SOURCES:.cu=.o)

# Final executable
TARGET = main

# Default target
all: $(TARGET)

# Link everything
$(TARGET): $(CC_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CC_OBJECTS) $(CU_OBJECTS) -o $(TARGET) -lcudart -L/usr/local/cuda/lib64

# Compile C++ files
%.o: %.cc $(HEADERS)
	$(CXX) -c $< -o $@

# Compile CUDA files
%.o: %.cu $(HEADERS)
	$(NVCC) -c $< -o $@

# Clean up object files and executable
clean:
	rm -f $(CC_OBJECTS) $(CU_OBJECTS) $(TARGET)
