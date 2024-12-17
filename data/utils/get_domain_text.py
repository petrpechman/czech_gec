import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str)
    parser.add_argument("--meta", type=str)
    parser.add_argument("--domain_meta", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--gold", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    domain = args.domain
    meta_filepath = args.meta
    domain_meta_file = args.domain_meta
    input_path = args.input
    gold_path = args.gold
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

    with open(input_path) as input_file, open(gold_path) as gold_file:
        input_lines = input_file.readlines()
        gold_lines = gold_file.readlines()

    domain = []
    for index, (input_line, gold_line) in enumerate(zip(input_lines, gold_lines)):
        if filter_for_data[index]:
            splitted_gold_line = gold_line.strip('\n').split('\t')
            splitted_gold_line = list(set(splitted_gold_line))
            for gold_line_part in splitted_gold_line:
                domain.append(gold_line_part.strip() + '\t' + input_line.strip())

    print('Size of domain:', len(domain))
    with open(output, 'w') as file:
        for sample in domain:
            file.write(sample + '\n')


if __name__ == "__main__":
    main()

# Example: python get_domain_text.py --domain 'Natives Formal' --meta ../geccc/data/meta.tsv --domain_meta ../geccc/data/train/sentence.meta --input ../geccc/data/train/sentence.input --gold ../geccc/data/train/sentence.gold --output output.tsv