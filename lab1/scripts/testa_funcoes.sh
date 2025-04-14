#!/bin/bash
#passa os parametros por meio da chamada, os dois parametros sao os valores de linha e coluna.

mkdir -p ../build

arquivo1="../arquivos/matrix1.dat"
arquivo2="../arquivos/matrix2.dat"

linhasM1="3200"
colunasM1="4000"
colunasM2="3200"

if [ $# -ge 2 ]; then
    linhasM1=$2
fi
if [ $# -ge 3 ]; then
    colunasM1=$3
fi
if [ $# -ge 4 ]; then
    colunasM2=$4
fi

../build/gera_matrix "$arquivo1" "$linhasM1" "$colunasM1"
../build/gera_matrix "$arquivo2" "$colunasM1" "$colunasM2"

gcc -Wall -std=c11 -mfma -o ../build/matrix_lib_test ../src/matrix_lib_test.c ../src/matrix_lib.c && 
../build/matrix_lib_test -s 5.0 -r $linhasM1 -c $colunasM1 -C $colunasM2 -m ../arquivos/matrix1.dat -M ../arquivos/matrix2.dat -o ../arquivos/result1.dat -O ../arquivos/result2.dat