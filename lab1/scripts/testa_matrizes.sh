#!/bin/bash

mkdir -p ../resultados

BINARIO="../build/matrix_lib_test"
ARQUIVOS="../arquivos"
CSV_RESULTADOS="../resultados/resultado.csv"

ESCALAR=5.0

echo "linhas_m1,colunas_m1,linhas_m2,colunas_m2,tempo_scalar_ms,tempo_matriz_ms,executavel" > "$CSV_RESULTADOS"

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


executaveis=(
    "../build/matrix_lib_test_intrisics_intrisics"
    "../build/matrix_lib_test_scalar_colunas"
    "../build/matrix_lib_test_scalar_linhas"
    "../build/matrix_lib_test_scalar_ptr"
    "../build/matrix_lib_test_mm_trad"
    "../build/matrix_lib_test_mm_opt"
)
for i in $(seq 1 5); do
    for exec in "${executaveis[@]}"; do
        for dimensao in "${dimensoes[@]}"; do
            set -- $dimensao
            linhas_m1=$1
            colunas_m1=$2
            linhas_m2=$colunas_m1
            colunas_m2=$linhas_m1

            matriz1="$ARQUIVOS/matrix_${linhas_m1}x${colunas_m1}.dat"
            matriz2="$ARQUIVOS/matrix_${linhas_m2}x${colunas_m2}.dat"
            result1="$ARQUIVOS/result1.dat"
            result2="$ARQUIVOS/result2.dat"

            echo "Executando teste com matrizes ${linhas_m1}x${colunas_m1} e ${linhas_m2}x${colunas_m2} para $exec..."

            output=$($exec -s "$ESCALAR" -r "$linhas_m1" -c "$colunas_m1" -C "$colunas_m2" -m "$matriz1" -M "$matriz2" -o "$result1" -O "$result2")

            tempo_scalar=$(echo "$output" | grep "Scalar product" | awk '{print $(NF-1)}')
            tempo_matriz=$(echo "$output" | grep "Matrix product" | awk '{print $(NF-1)}')
            if [ -z "$tempo_scalar" ]; then
                tempo_scalar="erro"
            fi
            if [ -z "$tempo_matriz" ]; then
                tempo_matriz="erro"
            fi

            echo "$linhas_m1,$colunas_m1,$linhas_m2,$colunas_m2,$tempo_scalar,$tempo_matriz,$(basename $exec)" >> "$CSV_RESULTADOS"
        done
    done
done

echo "Todos os testes foram finalizados. Resultados salvos em $CSV_RESULTADOS"
