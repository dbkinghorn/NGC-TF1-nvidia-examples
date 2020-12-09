import pandas as pd
import json
import glob
from tabulate import tabulate


glob_pattern = '*.json'

filenames = glob.glob(glob_pattern)
results = []
for filename in filenames:
    d = json.load(open(filename))
    r = {}

    fieldnames = ['Average Train Throughput', 'Best HR', 'Time to Best', 'seed', 'batch_size',
                  'num_gpus', 'AMP']
    colnames = ['average_train_throughput', 'average_accuracy', 'time_to_best_model', 'args_seed', 'args_batch_size',
                  'args_world_size', 'args_opt_level']

    field2col = dict(zip(fieldnames, colnames))

    try:
        for f, c in field2col.items():
            r[c] = d[f]
    except:
        pass
    results.append(r)

df = pd.DataFrame(results)
df.args_opt_level = df.args_opt_level.apply(lambda x: 'O2' if x == 1 else 'O0')
df = df.sort_values(['args_world_size', 'args_opt_level', 'args_seed'])
df['batch_size_per_gpu'] = df.args_batch_size / df.args_world_size
df = df.groupby(['args_opt_level', 'args_world_size', 'batch_size_per_gpu']).mean()

df1 = df[['best_accuracy', 'time_to_best_model']].unstack('args_opt_level')
df1['speedup'] = df1['time_to_best_model']['O0'] / df1['time_to_best_model']['O2']
df1 = df1.reset_index()

df1.loc[:, 'args_world_size'] = df1['args_world_size'].apply('{:,.0f}'.format)
df1.loc[:, 'batch_size_per_gpu'] = df1['batch_size_per_gpu'].apply('{:,.0f}'.format)
df1.loc[:, ('best_accuracy', 'O0')] = df1['best_accuracy']['O0'].apply('_{:,.5f}'.format).astype(str)
df1.loc[:, ('best_accuracy', 'O2')] = df1['best_accuracy']['O2'].apply('_{:,.5f}'.format).astype(str)

print(tabulate(df1, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f").replace('_', ' '))

######################################

df2 = df[['average_train_throughput']].unstack('args_opt_level')
df2['speedup'] = df2['average_train_throughput']['O2'] / df2['average_train_throughput']['O0']
df2['scaling_O0'] = df2['average_train_throughput']['O0'] / df2['average_train_throughput']['O0'].iloc[0]
df2['scaling_O2'] = df2['average_train_throughput']['O2'] / df2['average_train_throughput']['O2'].iloc[0]
df2 = df2.reset_index()

df2.loc[:, ('args_world_size')] = df2['args_world_size'].apply('{:,.0f}'.format)
df2.loc[:, ('batch_size_per_gpu')] = df2['batch_size_per_gpu'].apply('{:,.0f}'.format)
df2.loc[:, ('average_train_throughput', 'O2')] = df2['average_train_throughput']['O2'].apply('{:,.0f}'.format)
df2.loc[:, ('average_train_throughput', 'O0')] = df2['average_train_throughput']['O0'].apply('{:,.0f}'.format)

print()
print()
print()
print(tabulate(df2, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f").replace('_', ' '))
