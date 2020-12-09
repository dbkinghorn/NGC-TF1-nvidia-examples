import re
import json
import argparse

keys = [
        'batch_size',
        'num_gpus',
        'AMP',
        'Time to Best',
        'Time to Train',
        'seed',
        "Best HR",
        "Average Train Throughput",
        "Average Eval Throughput"
    ]

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
args = parser.parse_args()

text = open(args.filename).readlines()
text = '\n'.join(text)

result = {}
for field in keys:

    regex = '{}:\s*([0-9.]+)'.format(field)

    found = re.search(regex, text)
    if not found:
        continue

    val = float(found.groups()[0])
    result[field] = val

print(json.dumps(result, indent=4))


