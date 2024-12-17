import numpy as np

TOTAL_SIZES = [1000, 2500, 5000, 10000, 20000, 40000, 60000, 66673]
RANDOM_SEEDS = [14, 42, 74]
base = "../geccc/data/train/domains/"
folder = "../geccc/data/train/samples/"

# File paths
file_paths = {
    'natives_formal': base + 'natives_formal.tsv',
    'natives_web_informal': base + 'natives_web_informal.tsv',
    'romani': base + 'romani.tsv',
    'second_learners': base + 'second_learners.tsv'
}

# Sampling ratios for each file
ratio_s = [4060, 6977, 24824, 30812]
ratio = [(r / sum(ratio_s)) for r in ratio_s]

for total_size in TOTAL_SIZES:
    for random_seed in RANDOM_SEEDS:
        np.random.seed(random_seed)

        indices = np.random.choice(4, total_size, p=ratio)
        sizes = [len(indices[indices==i]) for i in [0, 1, 2, 3]]
        print(sizes)

        if sum(sizes) != total_size:
            raise Exception(ratio)

        lines_bucket = []
        for file_path in file_paths.values():
            with open(file_path) as f:
                lines = f.readlines()
            lines_bucket.append(lines)

        samples_bucket = []
        for i, lines in enumerate(lines_bucket):
            if sum(ratio_s) == total_size:
                samples = np.random.choice(lines, ratio_s[i], replace=False)
            else:
                samples = np.random.choice(lines, sizes[i], replace=False)
            samples_bucket.append(samples)

        combined_samples = np.concatenate(samples_bucket)
        np.random.shuffle(combined_samples)

        if len(combined_samples) != total_size:
            raise Exception(len(combined_samples))

        with open(folder + f'domain_sample_size_{total_size}_rs_{random_seed}.tsv', 'w') as f:
            for sample in combined_samples:
                f.write(sample)

        print(f"Sampling complete! Output saved as 'domain_sample_size_{total_size}_rs_{random_seed}.tsv'.")