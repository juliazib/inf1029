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
    "8 8"
    "32 32"
    "128 128"
    "512 512"
    "1024 1024"
    "2048 2048"
    "4096 4096"
    "8192 8192"
    "10000 10000"
    "512 1024"
    "1024 512"
    "2048 4096"
    "4096 2048"
    "8192 4096"
    "4096 8192"
)

for dimensao in "${dimensoes[@]}"; do
    set -- $dimensao 
    linhas=$1
    colunas=$2
    gerar_matriz "../arquivos/matrix_${linhas}x${colunas}.dat" "$linhas" "$colunas"
done
