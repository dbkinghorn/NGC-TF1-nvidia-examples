# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import subprocess


SSD_DIR = os.environ.get('SSD_DIR', '/workdir/models/research')

def extract_accuracy(output):
    tag = 'DetectionBoxes_Precision/mAP = '
    results = [line for line in output.splitlines() if tag in line]
    return float(re.search('DetectionBoxes_Precision/mAP = (\d*.\d*)', results[-1]).group(1))

def accuracy(precision, gpus, bs):
    gpus = gpus[:-3]
    precision = precision[2:]
    print(f'Testing mixed precision training accuracy on {gpus} GPUs')
    subprocess.run([
            'bash',
            f'{SSD_DIR}/examples/SSD320_FP{precision}_{gpus}GPU_BENCHMARK.sh',
            f'/results/SSD320_FP{precision}_{gpus}GPU',
            f'{SSD_DIR}/configs'])
    with open(f'/results/SSD320_FP{precision}_{gpus}GPU/train_log') as f:
        output = f.read()
    print(output)
    return extract_accuracy(output)

def convergence(precision, gpus, bs):
    gpus = gpus[:-3]
    precision = precision[2:]
    print(f'Testing mixed precision training convergence on {gpus} GPUs')
    output = subprocess.run([
            'bash',
            f'{SSD_DIR}/examples/SSD320_FP{precision}_{gpus}GPU.sh',
            f'/results/SSD320_FP{precision}_{gpus}GPU',
            f'{SSD_DIR}/configs'],
        encoding='utf8',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    print(output)
    return extract_accuracy(output)

def mini_infer_bench(*args, **kwargs):
    return infer_bench(*args, **kwargs)

def infer_bench(precision, gpus, bs):
    gpus = gpus[:-3]
    precision = precision[2:]
    bs = bs[2:]
    print(f'Testing mixed precision inference performance on {gpus} GPUs using batch size = {bs}')
    output = subprocess.run([
            'bash',
            f'{SSD_DIR}/examples/SSD320_FP{precision}_inference.sh',
            f'{SSD_DIR}/configs',
            '--batch_size',
            str(bs)],
        encoding='utf8',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    print(output)
    def extract_result(output):
        return float(output.splitlines()[-1].split()[2])
    return extract_result(output)

def mini_train_bench(*args, **kwargs):
    return train_bench(*args, **kwargs)

def train_bench(precision, gpus, bs):
    gpus = gpus[:-3]
    precision = precision[2:]
    print(f'Testing mixed precision training speed on {gpus} GPUs')
    output = subprocess.run([
            'bash',
            f'{SSD_DIR}/examples/SSD320_FP{precision}_{gpus}GPU_BENCHMARK.sh',
            f'/results/SSD320_FP{precision}_{gpus}GPU',
            f'{SSD_DIR}/configs'],
        encoding='utf8',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    with open(f'/results/SSD320_FP{precision}_{gpus}GPU/train_log') as f:
        log = f.read()
        print(log)
    print(output)
    def extract_result(output):
        return float(output.splitlines()[-1].split()[-2])
    return extract_result(output)

