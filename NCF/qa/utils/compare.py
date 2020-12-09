import re
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--reference', type=str, required=True)
parser.add_argument('--tolerance', type=float, default=0.9)
parser.add_argument('--keys', type=str, default=None)

args = parser.parse_args()

with open(args.file, 'r') as result_file:
    data = result_file.read()
with open(args.reference, 'r') as ref_file:
    ref_data = ref_file.read()


if not args.keys:
    args.keys = [
        "Best HR",
        "Average Train Throughput",
        "Average Eval Throughput"
    ]
else:
    args.keys = args.keys.split(',')

fail = False
for field in args.keys:
    print('Checking: ', field)
    regex = '{}:\s*([0-9.]+)'.format(field)

    val = float(re.search(regex, data).groups()[0])
    ref = float(re.search(regex, ref_data).groups()[0])
    
    if val < args.tolerance * ref:
        print("FAILED at {}".format(field))
        fail = True
    else:
        print("PASSED")

if fail:
    sys.exit(1)
else:
    sys.exit(0)
