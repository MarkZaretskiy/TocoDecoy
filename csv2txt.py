import pandas as pd
import argparse

def write_txt(smiles, names, fname):
    with open(fname, "w") as fout:
        for s, n in zip(smiles, names):
            line = f"{s} {n}\n"
            fout.write(line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--json", type=int)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.json:
        df = pd.read_json(args.input) 
    else:
        df = pd.read_csv(args.input)

    smiles = df['smiles'].values
    names = [i for i in range(len(smiles))]

    write_txt(smiles, names, args.output)