# Add invlib to PYTHONPATH
export PYTHONPATH=/home/olemar/MISU/MATS/invlib:${PYTHONPATH}
# Load MKL
source ~/intel/mkl/bin/mklvars.sh intel64

# Force MKL preload to avoid undefined symbol problem.
export LD_PRELOAD=/home/olemar/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin/libmkl_core.so:/home/olemar/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin/libmkl_intel_thread.so:/home/olemar/intel/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64/libiomp5.so


