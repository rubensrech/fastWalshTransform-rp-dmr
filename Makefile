TARGET=fastWalshTransform
SRC_DIR=.
OBJ_DIR=./obj

CUDAPATH=/usr/local/cuda
NVCC=nvcc
NVCCFLAGS= -std=c++11 -O3 -Xptxas -v
ARCH= -gencode arch=compute_60,code=sm_60 \
	  -gencode arch=compute_61,code=sm_61 \
	  -gencode arch=compute_62,code=[sm_62,compute_62] \
	  -gencode arch=compute_70,code=[sm_70,compute_70]

CPP=g++
CXXFLAGS= -std=c++11 -o2 -fPIC -fopenmp 
LDFLAGS= -L$(CUDAPATH)/lib64  -lcudart  -lcurand
INCLUDE= -I$(CUDAPATH)/include

CPP_FILES=$(wildcard *.cpp)
CU_FILES=$(wildcard *.cu)
H_FILES=$(wildcard *.h)
CUH_FILES=$(wildcard *.cuh)
OBJ_FILES=$(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES=$(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.o)))
OBJS=$(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS+=$(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

all: mkdirobj $(TARGET)

test:
	echo $(OBJS)

mkdirobj:
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CPP) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(CUH_FILES)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(H_FILES)
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

clean:
	rm -rf $(OBJ_DIR)
	rm $(TARGET)