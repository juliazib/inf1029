#!/bin/bash

mkdir -p ../arquivos

BINARIO="../build/matrix_lib_test"
ARQUIVOS="../arquivos"
CSV_RESULTADOS_ESCALAR="../resultados/resultadoEscalar.csv"
CSV_RESULTADOS_MATRIZ_MATRIZ="../resultados/resultadoMatrizMatriz.csv"

ESCALAR=5.0

echo "linhas_m1,colunas_m1,tempo_scalar_ms,executavel" > "$CSV_RESULTADOS_ESCALAR"
echo "linhas_m1,colunas_m1,linhas_m2,colunas_m2,tempo_matriz_matriz_ms,executavel" > "$CSV_RESULTADOS_MATRIZ_MATRIZ"

dimensoes=(
    "8 8"
    "32 32"
    "128 128"
    "512 512"
)

executaveisMatrizesMatrizes=(
    "../build/matrix_lib_test_intrisics_intrisics"
    "../build/matrix_lib_test_mm_trad"
    "../build/matrix_lib_test_mm_opt"
)

executaveisEscalares=(
    "../build/matrix_lib_test_intrisics_intrisics"
    "../build/matrix_lib_test_scalar_colunas"
    "../build/matrix_lib_test_scalar_linhas"
    "../build/matrix_lib_test_scalar_ptr"
)

# EXECUTA PARA ESCALARES
for exec in "${executaveisEscalares[@]}"; do
    for dimensao in "${dimensoes[@]}"; do
        for i in $(seq 1 5); do
            set -- $dimensao
            linhas_m1=$1
            colunas_m1=$2
            linhas_m2=$colunas_m1
            colunas_m2=$linhas_m1

            matriz1="$ARQUIVOS/matrix_${linhas_m1}x${colunas_m1}.dat"
            matriz2="$ARQUIVOS/matrix_${linhas_m2}x${colunas_m2}.dat"
            result1="$ARQUIVOS/result1.dat"
            result2="$ARQUIVOS/result2.dat"

            echo "Executando teste com matrizes ${linhas_m1}x${colunas_m1} para $(basename $exec)..."

            output=$($exec -s "$ESCALAR" -r "$linhas_m1" -c "$colunas_m1" -C "$colunas_m2" -m "$matriz1" -M "$matriz2" -o "$result1" -O "$result2")

            tempo_scalar=$(echo "$output" | grep "Scalar product" | awk '{print $(NF-1)}')
            if [ -z "$tempo_scalar" ]; then
                tempo_scalar="erro"
            fi
            echo "$linhas_m1,$colunas_m1,$tempo_scalar,$(basename $exec)" >> "$CSV_RESULTADOS_ESCALAR"
        done
    done
done

#EXECUTA PARA MULTIPLICACAO DE MATRIZES
for i in $(seq 1 5); do
    for exec in "${executaveisMatrizesMatrizes[@]}"; do
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

            echo "Executando teste com matrizes ${linhas_m1}x${colunas_m1} e ${linhas_m2}x${colunas_m2} para $(basename $exec)..."

            output=$($exec -s "$ESCALAR" -r "$linhas_m1" -c "$colunas_m1" -C "$colunas_m2" -m "$matriz1" -M "$matriz2" -o "$result1" -O "$result2")

            tempo_matriz=$(echo "$output" | grep "Matrix product" | awk '{print $(NF-1)}')

            if [ -z "$tempo_matriz" ]; then
                tempo_matriz="erro"
            fi

            echo "$linhas_m1,$colunas_m1,$linhas_m2,$colunas_m2,$tempo_matriz,$(basename $exec)" >> "$CSV_RESULTADOS_MATRIZ_MATRIZ"
        done
    done
done

echo "Todos os testes foram finalizados. Resultados salvos em $CSV_RESULTADOS"

