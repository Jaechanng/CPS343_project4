#!/bin/bash

MATRIX_DIR="/home/cps343/matrix"

PROGRAMS=(
    "./proj4-cublas"
    "./proj4-cuda"
)

MATRICES=(
    "A-5000x5000.dat"
    "A-10000x10000.dat"
)

echo "======================================"
echo " CUDA Benchmark (Raw Output)"
echo "======================================"

for prog in "${PROGRAMS[@]}"; do
    for matrix in "${MATRICES[@]}"; do

        MATRIX_PATH="${MATRIX_DIR}/${matrix}"

        echo ""
        echo "--------------------------------------"
        echo "Program : $prog"
        echo "Matrix  : $matrix"
        echo "--------------------------------------"

        echo ""
        echo "[P1000 ]"
        srun --gres=gpu:P1000 $prog $MATRIX_PATH

        echo ""
        echo "[RTX3000]"
        srun --gres=gpu:RTX3000 $prog $MATRIX_PATH

        echo ""
        echo "[RTXA5000 - ml partition]"
        srun -p ml --gres=gpu:RTXA5000 $prog $MATRIX_PATH

        echo ""
    done
done

echo "======================================"
echo " Done"
echo "======================================"