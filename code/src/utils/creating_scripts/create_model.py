import argparse

from pathlib import Path
from transformers import BartConfig
from transformers import TFAutoModelForSeq2SeqLM

config = BartConfig(
                vocab_size=32_000,
                max_position_embeddings=256,
                encoder_layers=6,
                encoder_ffn_dim=2048,
                encoder_attention_heads=8,
                decoder_layers=6,
                decoder_ffn_dim=2048,
                decoder_attention_heads=8,
                encoder_layerdrop=0.0,
                decoder_layerdrop=0.0,
                activation_function="relu",
                d_model=512,
                dropout=0.1,
                attention_dropout=0.0,
                activation_dropout=0.0,
                init_std=0.02,
                classifier_dropout=0.0,
                scale_embedding=True,
                use_cache=True,
                num_labels=3,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                is_encoder_decoder=True,
                decoder_start_token_id=1,
                forced_eos_token_id=2,
            )

config_big = BartConfig(
                vocab_size=32_000,
                max_position_embeddings=256,
                encoder_layers=6,
                encoder_ffn_dim=4096,
                encoder_attention_heads=16,
                decoder_layers=6,
                decoder_ffn_dim=4096,
                decoder_attention_heads=16,
                encoder_layerdrop=0.0,
                decoder_layerdrop=0.0,
                activation_function="relu",
                d_model=1024,
                dropout=0.3,
                attention_dropout=0.0,
                activation_dropout=0.0,
                init_std=0.02,
                classifier_dropout=0.0,
                scale_embedding=True,
                use_cache=True,
                num_labels=3,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                is_encoder_decoder=True,
                decoder_start_token_id=1,
                forced_eos_token_id=2,
            )

config_small = BartConfig(
                vocab_size=32_000,
                max_position_embeddings=256,
                encoder_layers=2,
                encoder_ffn_dim=1024,
                encoder_attention_heads=4,
                decoder_layers=2,
                decoder_ffn_dim=1024,
                decoder_attention_heads=4,
                encoder_layerdrop=0.0,
                decoder_layerdrop=0.0,
                activation_function="relu",
                d_model=256,
                dropout=0.1,
                attention_dropout=0.0,
                activation_dropout=0.0,
                init_std=0.02,
                classifier_dropout=0.0,
                scale_embedding=True,
                use_cache=True,
                num_labels=3,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                is_encoder_decoder=True,
                decoder_start_token_id=1,
                forced_eos_token_id=2,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    
    args.output_dir = args.output_dir.rstrip('/') + '/'
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    model = TFAutoModelForSeq2SeqLM.from_config(config)
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()