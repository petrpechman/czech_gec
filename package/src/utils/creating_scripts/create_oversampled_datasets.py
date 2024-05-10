import math
import random
import argparse

from typing import List

random.seed(42)

def get_lines(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nf", type=str)
    parser.add_argument("--nwi", type=str)
    parser.add_argument("--romani", type=str)
    parser.add_argument("--sl", type=str)

    parser.add_argument("-p", type=float)
    parser.add_argument("-t", type=int)

    parser.add_argument("-o", type=str, default='output.tsv')

    args = parser.parse_args()
    
    nf_lines = get_lines(args.nf)
    nwi_lines = get_lines(args.nwi)
    romani_lines = get_lines(args.romani)
    sl_lines = get_lines(args.sl)

    nf_size = math.pow(len(nf_lines) * 1.0, args.p) 
    nwi_size = math.pow(len(nwi_lines) * 1.0, args.p) 
    romani_size = math.pow(len(romani_lines) * 1.0, args.p) 
    sl_size = math.pow(len(sl_lines) * 1.0, args.p)
    
    print("Sizes:")
    print(f"NF: {nf_size}")
    print(f"NWI: {nwi_size}")
    print(f"ROMANI: {romani_size}")
    print(f"SL: {sl_size}")

    total = nf_size + nwi_size + romani_size + sl_size

    nf_prob = nf_size / total
    nwi_prob = nwi_size / total
    romani_prob = romani_size / total
    sl_prob = sl_size / total
    
    print("Probs:")
    print(f"NF: {nf_prob}")
    print(f"NWI: {nwi_prob}")
    print(f"ROMANI: {romani_prob}")
    print(f"SL: {sl_prob}")

    nf_border = nf_prob
    nwi_border = nwi_prob + nf_border
    romani_border = romani_prob + nwi_border
    sl_border = sl_prob + romani_border

    print("Borders:")
    print(f"NF: {nf_border}")
    print(f"NWI: {nwi_border}")
    print(f"ROMANI: {romani_border}")
    print(f"SL: {sl_border}")

    result_lines = []

    for i in range(args.t):
        p = random.random()
        if p < nf_border:
            line = random.choice(nf_lines)
        elif p < nwi_border:
            line = random.choice(nwi_lines)
        elif p < romani_border:
            line = random.choice(romani_lines)
        else:
            line = random.choice(sl_lines)
        
        result_lines.append(line)

    with open(args.o, 'w') as f:
        for line in result_lines:
            f.write(line)


if __name__ == "__main__":
    main()

# Example: python create_oversampled_datasets.py --nf natives_formal.tsv --nwi natives_web_informal.tsv --romani romani.tsv --sl second_learners.tsv -p 0.5 -t 20000