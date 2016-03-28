set -x
rm -rf ls ~/.theano/*

export MKL_NUM_THREADS=1
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
export OMP_NUM_THREADS=32
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"

export THEANO_FLAGS=mode=FAST_RUN,openmp=True,device=cpu,floatX=float32,cxx=cnn.sh

for N in  1 2 4 8 16 32
do
echo OMP_NUM_THREADS=$N  python conv.py
echo "=========================================="
OMP_NUM_THREADS=$N  python conv.py
done
