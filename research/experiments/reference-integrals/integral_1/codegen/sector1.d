SECTOR1_CPP = \
	src/sector_1.cpp \
	src/sector_1_0.cpp
SECTOR1_DISTSRC = \
	distsrc/sector_1_0.cpp \
	distsrc/sector_1_0.cu
SECTOR1_MMA = \
	mma/sector_1_0.m
SECTOR_CPP += $(SECTOR1_CPP)
SECTOR_MMA += $(SECTOR1_MMA)

$(SECTOR1_DISTSRC) $(SECTOR1_CPP) $(patsubst %.cpp,%.hpp,$(SECTOR1_CPP)) : codegen/sector1.done ;
$(SECTOR1_MMA) : codegen/sector1.mma.done ;
