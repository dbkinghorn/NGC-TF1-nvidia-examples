#!/bin/bash

set -e
set -x

platform=dgx1v32g
precision=fp32

./qa/utils/prepare_qa_dataset.sh

mpirun -np 8 --allow-run-as-root python ncf.py --verbose --data /ncf_data/cache/ml-20m --checkpoint-dir /ncf_data/checkpoints --fp16 | tee nv.log

python qa/utils/compare.py --file nv.log --reference qa/baseline/$platform.$precision.8.gpu.log
