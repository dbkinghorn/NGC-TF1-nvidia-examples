#!/bin/bash

set -e
set -x

platform=t4
precision=fp32

if [ -d /data/ncf/ml-20m ]; then
    DATADIR=/data/ncf/ml-20m
else
    ./qa/utils/prepare_qa_dataset.sh
    DATADIR=/ncf_data/cache/ml-20m
fi

mpirun -np 1 --allow-run-as-root python ncf.py --verbose --data ${DATADIR} --mode test | tee nv.log

python qa/utils/compare.py --file nv.log --reference qa/baseline/$platform.$precision.1.gpu.log --keys "Average Eval Throughput"
