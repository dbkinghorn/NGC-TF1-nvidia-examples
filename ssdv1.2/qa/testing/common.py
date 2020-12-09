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

import math
import yaml


def load_baselines(baselines):
    with open(baselines, 'r') as f:
        return yaml.safe_load(f)


def evaluate_result(baseline, result):
    failed = False
    message = f'{baseline["unit"]}: {result} expected {baseline["unit"]}: {baseline["target"]}'
    if 'lower_tolerance' in baseline:
        lower_bound = baseline['target'] * (1 - baseline['lower_tolerance'])
        message += f' accepted minimum: {lower_bound}'
        failed = failed or math.isnan(result) or result < lower_bound
    if 'upper_tolerance' in baseline:
        upper_bound = baseline['target'] * (1 + baseline['upper_tolerance'])
        message += f' accepted maximum: {upper_bound}'
        failed = failed or math.isnan(result) or result > upper_bound
    print(message)
    print('FAILED' if failed else 'PASSED')
    return failed


def evaulate_benchmark(benchmark, baselines, benchmark_args=[]):
    if 'target' in baselines or 'tolerance' in baselines:
        assert({ 'target', 'unit' }.issubset(baselines.keys()))
        result = benchmark(*benchmark_args)
        new_baseline = baselines.copy()
        new_baseline['target'] = result
        return evaluate_result(baselines, result), new_baseline
    else:
        failed = False
        results = {}
        for param in baselines:
            test_failed, result = evaulate_benchmark(benchmark, baselines[param], [*benchmark_args, param])
            failed |= test_failed
            results[param] = result
        return failed, results


