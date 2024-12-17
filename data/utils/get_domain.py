import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str)
    parser.add_argument("--meta", type=str)
    parser.add_argument("--domain_meta", type=str)
    parser.add_argument("--domain_m2", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    domain = args.domain
    meta_filepath = args.meta
    domain_meta_file = args.domain_meta
    domain_m2_file = args.domain_m2
    output = args.output

    data = pd.read_csv(meta_filepath, sep='\t')
    data = data[['Filename', 'Domain']]
    filenames = data[data['Domain'] == domain]['Filename']
    filenames = list(set(filenames.to_list()))

    with open(domain_meta_file, 'r') as file:
        files = file.readlines()

    filter_for_data = []
    for file in files:
        file = file.strip()
        if file in filenames:
            filter_for_data.append(True)
        else:
            filter_for_data.append(False)

    with open(domain_m2_file, 'r') as file:
        lines = file.readlines()

    lines = "".join(lines)
    lines = lines.split('\n\n')
    lines = [line.strip() for line in lines]

    domain = []
    for i, line in enumerate(lines):
        if filter_for_data[i]:
            domain.append(line)

    print('Size of domain:', len(domain))
    with open(output, 'w') as file:
        for sample in domain[:-1]:
            file.write(sample + '\n')
            file.write('\n')
        file.write(domain[-1] + '\n')


if __name__ == "__main__":
    main()

# Example: python get_domain.py --domain 'Natives Formal' --meta ../geccc/meta.tsv --domain_meta ../geccc/dev/sentence.meta --domain_m2 ../geccc/dev/sentence.m2 --output output.m2