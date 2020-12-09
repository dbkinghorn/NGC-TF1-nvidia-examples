import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
import pandas as pd
from tabulate import tabulate

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 100000)
pd.options.display.float_format = '{:,.3f}'.format

filenames = glob.glob('inference*.json')
filenames = sorted(filenames)


def latency_to_throughput(latency_df):
    throughput_df_names = []
    for c in latency_df:
        if 'latency' not in c or 'speedup' in c:
            throughput_df_names.append(c)
            continue
        name = c.replace('latency', 'throughput')
        milliseconds_in_a_second = 1000
        latency_df[name] = milliseconds_in_a_second * latency_df.batch_size / latency_df[c]
        throughput_df_names.append(name)
    throughput_df = latency_df[throughput_df_names]
    return throughput_df

warmup = 20
def latency_table(data, qs, warmup=20):
    latencies = sorted(data['inference_latencies'][warmup:])
    result = OrderedDict()
    result['average_latency'] = np.mean(latencies)
    for q in qs:
        result['latency_{}_confidence'.format(q)] = np.percentile(latencies, q)
    return result

bins = np.arange(0.3, 0.6, 0.001)
fig, axes = plt.subplots(len(filenames), 1, figsize=(15, 3 * len(filenames)), sharex=True)
table_data = []

for i, f in enumerate(filenames):
    ax = axes[i]
    data = json.load(open(f))

    t = latency_table(data, qs=[50, 90, 95, 99, 100], warmup=warmup)
    t['precision'] = 'FP16' if data['args']['amp'] else 'FP32'
    t['batch_size'] = data['args']['batch_size']
    table_data.append(t)

    latencies_ms = np.array(data['inference_latencies'][warmup:]) * 1000
    _ = ax.hist(latencies_ms, bins=bins, label=os.path.basename(f))
    ax.legend()
    ax.grid()
    ax.set_ylabel('# experiments')

plt.xlabel('latency [ms]')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('latencies.pdf')

df = pd.DataFrame(table_data)
for c in df.columns:
    if 'latency' in c:
        df[c] = df[c] * 1000

columns = ["batch_size", "precision", "average_latency", "latency_50_confidence",
           "latency_90_confidence", "latency_95_confidence",
           "latency_99_confidence", "latency_100_confidence"]
df = df[columns]
df = df.sort_values(by=['batch_size', 'precision'], ascending=True)


print()
print('============= LATENCY ==============')
print()

df_fp32 = df.loc[df.precision == 'FP32',:]
df_fp32 = df_fp32.reset_index(drop=True)
print(tabulate(df_fp32, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f").replace('_', ' '))
print()

df_fp16 = df.loc[df.precision == 'FP16',:]
df_fp16 = df_fp16.reset_index(drop=True)

cols = [c for c in df_fp16.columns if 'latency' in c]
speedup_df = df_fp32[cols] / df_fp16[cols]
speedup_df.columns = [c + '_speedup' for c in speedup_df.columns]

df_fp16 = pd.concat([df_fp16, speedup_df], axis=1)
columns_fp16 = []
for c in columns:
    columns_fp16.append(c)
    if 'latency' in c:
        columns_fp16.append(c + '_speedup')
df_fp16 = df_fp16[columns_fp16]

print(tabulate(df_fp16, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f").replace('_', ' '))

print()
print('============= THROUGHPUT ==============')
print()

throughput_fp32 = latency_to_throughput(df_fp32)
throughput_fp16 = latency_to_throughput(df_fp16)

print(tabulate(throughput_fp32, headers='keys', tablefmt='pipe', showindex=False, floatfmt=",.3f").replace('_', ' '))
print()
print(tabulate(throughput_fp16, headers='keys', tablefmt='pipe', showindex=False, floatfmt=",.3f").replace('_', ' '))

