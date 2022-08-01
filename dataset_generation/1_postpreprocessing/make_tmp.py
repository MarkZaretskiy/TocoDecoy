import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


# df.iloc[:, len(df.columns)] = 
new_lines = []
with open(args.input, "r") as fin:
    lines = fin.readlines()

for l in lines:
    if l != "<EOS>\n":
        values = l.split()
        values.insert(8, "0")
        new_l = " ".join(values)
    else:
        new_l = l
    new_lines.append(new_l)

with open(args.output, "w") as fout:
    for l in new_lines:
        fout.write(l)
        fout.write("\n")

