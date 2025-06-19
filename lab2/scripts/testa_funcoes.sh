#!/bin/bash
#passa os parametros por meio da chamada, os dois parametros sao os valores de linha e coluna.

mkdir -p ../build && gcc  -Wall -o ../build/gera_matrix ../src/gera_matrix.c

arquivo1="../arquivos/matrix1.dat"
arquivo2="../arquivos/matrix2.dat"

linhasM1="3200"
colunasM1="4000"
colunasM2="3200"

if [ $# -ge 1 ]; then
    linhasM1=$1
fi
if [ $# -ge 2 ]; then
    colunasM1=$2
fi
if [ $# -ge 3 ]; then
    colunasM2=$3
fi

echo $linhasM1 $colunasM1 $colunasM2

nvcc -o ../build/matrix_lib_test ../src/matrix_lib_test.cu ../src/matrix_lib.cu && 
../build/matrix_lib_test \
    -s 5.0 \
    -r $linhasM1 \
    -c $colunasM1 \
    -C $colunasM2 \
    -m ../arquivos/matrix1.dat \
    -M ../arquivos/matrix2.dat \
    -o ../arquivos/result1.dat \
    -O ../arquivos/result2.dat \
    -t 256 \
    -g 4096