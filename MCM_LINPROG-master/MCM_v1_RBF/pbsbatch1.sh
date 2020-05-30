#!/bin/bash
#PBS -N MCM_RBF
#PBS -P ee
#PBS -m bea
#PBS -M eez142368@iitd.ac.in
#PBS -l select=1:ncpus=1
#PBS -o stdout_file1
#PBS -e stderr_file1
#PBS -l walltime=160:00:00
#PBS -l matlab=1
#PBS -V
#PBS -l software=MATLAB
cd $PBS_O_WORKDIR
module load apps/matlab
time -p matlab -nosplash -nodisplay <test_bench_EFS_SVM_v1.m;exit> LMM1.log