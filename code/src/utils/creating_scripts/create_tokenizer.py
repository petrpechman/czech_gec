import argparse
import tensorflow as tf

from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=5)
    args = parser.parse_args()
    
    args.output_dir = args.output_dir.rstrip('/') + '/'

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    tokenizer.train(
        args.corpus_file,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )

    tokenizer.save_model(args.output_dir)
    tokenizer.save(args.output_dir + "tokenizer.json")

if __name__ == "__main__":
    main()

    # Example how to use:
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("../../models/transformer") # output-dir 
    # decoded = tokenizer.decode(tokenizer.encode("Monkey D. Luffy"), skip_special_tokens=True)
    # print(decoded)

# CMD: python3 tokenizer_utils.py --corpus-file ../../data/tokenized/news-2018-cs-tokenized.txt --output-dir transformer/model/
