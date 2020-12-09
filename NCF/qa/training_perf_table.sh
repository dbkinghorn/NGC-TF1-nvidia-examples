#! /bin/bash
set -e
set -x

./prepare_dataset.sh

for gpus in 1 8; do
    for precision in FP16 FP32; do
        for seed in $(seq 1 2); do
            echo "num gpus: ${gpus}, precision: ${precision}"
            if [ ${precision}  == "FP16" ]; then
                mpirun -np ${gpus} --allow-run-as-root python ncf.py --seed ${seed} --data /data/cache/ml-20m --fp16 | tee nv.log
            else
                mpirun -np ${gpus} --allow-run-as-root python ncf.py --seed ${seed} --data /data/cache/ml-20m | tee nv.log
            fi

            python ./qa/utils/logs_parser.py --file nv.log > training_${gpus}_${precision}_${seed}.json
        done
    done
done

pip install tabulate
python qa/utils/train_perf_table_analysis.py
