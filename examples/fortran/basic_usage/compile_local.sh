NVARCH=Linux_x86_64; export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS
MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/25.1/compilers/man; export MANPATH
PATH=$NVCOMPILERS/$NVARCH/25.1/compilers/bin:$PATH; export PATH
export PATH=$NVCOMPILERS/$NVARCH/25.1/comm_libs/mpi/bin:$PATH
export MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/25.1/comm_libs/mpi/man
LD_LIBRARY_PATH=/home/milton/MHIT36_cuDecomp/cuDecomp-main/build/include
make clean
make

