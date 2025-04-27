#!/bin/bash

mkdir -p ../build
gcc -Wall -o ../build/gera_matrix ../src/gera_matrix.c

gerar_matriz() {
    arquivo=$1
    linhas=$2
    colunas=$3
    echo "Gerando matriz $arquivo com dimensões ${linhas}x${colunas}..."
    ../build/gera_matrix "$arquivo" "$linhas" "$colunas"
}

mkdir -p ../arquivos

dimensoes=(
    "32 32"
    "1024 1024"
    "8192 8192"
    "16384 16384"
    "32768 32768"
    "512 1024"
    "1024 512"
    "8192 4096"
    "4096 8192"
    "32000 40000"
    "40000 32000"
)

for dimensao in "${dimensoes[@]}"; do
    set -- $dimensao 
    linhas=$1
    colunas=$2
    gerar_matriz "../arquivos/matrix_${linhas}x${colunas}.dat" "$linhas" "$colunas"
done
