# ERROR_METRIC = uint_error | relative_error | hybrid
ERROR_METRIC=hybrid
FIND_THRESHOLD=0

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

ifeq ($(ERROR_METRIC), relative_error) 
CXXFLAGS+= -DERROR_METRIC=0
NVCCFLAGS+= -DERROR_METRIC=0
endif

ifeq ($(ERROR_METRIC), uint_error) 
CXXFLAGS+= -DERROR_METRIC=1
NVCCFLAGS+= -DERROR_METRIC=1
endif

ifeq ($(ERROR_METRIC), hybrid) 
CXXFLAGS+= -DERROR_METRIC=2
NVCCFLAGS+= -DERROR_METRIC=2
endif

ifeq ($(FIND_THRESHOLD), 1) 
CXXFLAGS+= -DFIND_THRESHOLD
NVCCFLAGS+= -DFIND_THRESHOLD
endif

all: mkdirobj $(TARGET)

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

copy_titanV:
	scp *.{cu,h,cpp} gpu_carol_titanV:rubens/fastWalshTransform-dmr
	scp Makefile gpu_carol_titanV:rubens/fastWalshTransform-dmr

copy_nvbitfi_titanV:
	# scp *.{cu,h,cpp,sh} gpu_carol_titanV:rubens/nvbitfi/test-apps/fastWalshTransform-dmr-rp
	scp Makefile gpu_carol_titanV:rubens/nvbitfi/test-apps/fastWalshTransform-dmr-rp

copy_p100:
	scp *.{cu,h,cpp} gppd:fastWalshTransform-dmr
	# scp Makefile gppd:fastWalshTransform-dmr

test:
	./fastWalshTransform -input inputs/input-bit-21.data -measureTime 1

golden:
	./fastWalshTransform -input inputs/input-bit-21.data > golden_stdout.txt 2> golden_stderr.txt
	mv out-vs-gold-stats.txt golden_out-vs-gold-stats.txt