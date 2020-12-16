import sys
import os
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 %s <filename>" % sys.argv[0])
    exit(-1)

filename = sys.argv[1]

df = pd.read_csv(filename)

df['relative_sum'] = df['avg_err'] * df['diff_vals_count']
df['benchmark'] = 'fastWalshTransform'

cols = ['benchmark', 'error_model', 'detection_outcome', 'relative_sum']
df = df[cols]

outFilename = os.path.dirname(filename) + '/sdcs-relative-sum.csv'

df.to_csv(outFilename, index=False)
