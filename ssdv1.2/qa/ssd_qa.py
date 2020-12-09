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

import os
import argparse
import yaml

import testing as tst
import testing.ssd as ssd_tst


SSD_DIR = os.environ.get('SSD_DIR', '/workdir/models/research')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', default=f'{SSD_DIR}/qa/baselines.yaml')
    parser.add_argument('--benchmark')
    parser.add_argument('--hardware', default='DGX1V')
    parser.add_argument('--precision')
    return parser

def main():
    args = get_parser().parse_args()
    baselines = tst.load_baselines(args.baselines)
    failed, results = tst.evaulate_benchmark(
                getattr(ssd_tst, args.benchmark),
                baselines[args.benchmark][args.hardware][args.precision],
                benchmark_args=[args.precision])
    print('baselines:')
    print(yaml.dump({ args.benchmark: { args.hardware: { args.precision: baselines[args.benchmark][args.hardware][args.precision] } } }))
    print('benchmark results:')
    print(yaml.dump({ args.benchmark: { args.hardware: { args.precision: results } } }))
    print('final result:', 'FAILED' if failed else 'PASSED')
    exit(1 if failed else 0)

if __name__ == '__main__':
    main()
