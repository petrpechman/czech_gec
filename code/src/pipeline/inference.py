import sys
sys.path.append('..')

import os
import time
import numpy as np
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import dataset_utils
from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)
    ### Params:
    num_beams = 4
    min_length = 0
    length_penalty = 1.0
    ###
    
    SEED = config['seed']

    # data loading
    M2_DATA_DEV = config['m2_data_dev']
    if not isinstance(M2_DATA_DEV, str):
        M2_DATA_DEV, _ = M2_DATA_DEV

    M2_DATA_TEST = config['m2_data_test']
    if not isinstance(M2_DATA_TEST, str):
        M2_DATA_TEST, _ = M2_DATA_TEST
    DEV_GECCC_DATASETS = config.get('dev_geccc_datasets', [])
    TEST_GECCC_DATASETS = config.get('test_geccc_datasets', [])
    RETAG_DEV_GECCC_DATASETS = config.get('retag_dev_geccc_datasets', [])
    RETAG_TEST_GECCC_DATASETS = config.get('retag_test_geccc_datasets', [])
    OTHER_DATASETS = config.get('other_datasets', [])
    BATCH_SIZE = config['batch_size']
    
    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config']
    USE_F16 = False
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    
    MAX_EVAL_LENGTH = config['max_eval_length']
    FILE_DEV_PREDICTIONS = 'predictions_dev.txt'
    FILE_TEST_PREDICTIONS = 'predictions_test.txt'

    EVAL_GECCC_EVERY = config.get('eval_geccc_every', 10)

    tf.random.set_seed(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    
    ### Dataset loadings:
    def get_tokenized_sentences(line):
        # only tokenize line
        line = line.decode('utf-8')
        tokenized = tokenizer(line, max_length=MAX_EVAL_LENGTH, truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask']

    def tokenize_line(line):
        # wrapper for tokenize_line
        input_ids, attention_mask = tf.numpy_function(get_tokenized_sentences, inp=[line], Tout=[tf.int32, tf.int32])
        dato = {'input_ids': input_ids[0],
                'attention_mask': attention_mask[0]}
        return dato
    
    def get_dataset_pipeline(source_sentences) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((source_sentences))
        dataset = dataset.map(tokenize_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={'input_ids': [None], 'attention_mask': [None]})
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    
    dev_source_sentences, _ = load_annotation(M2_DATA_DEV)
    test_source_sentences, _ = load_annotation(M2_DATA_TEST)
    
    datasets = []
    refs = []
    eval_types = []
    for dataset in OTHER_DATASETS:
        if not isinstance(dataset, str):
            dataset, eval_type = dataset
        else:
            eval_type = ['m2_scorer'] 
        source_sentences, gold_edits = load_annotation(dataset)
        if 'errant' in eval_type:
            ref_m2 = open(dataset).read().strip().split("\n\n")
            refs.append(ref_m2)
        else:
            refs.append(None)
        eval_types.append(eval_type)
        datasets.append((source_sentences, gold_edits, dataset))

    dev_dataset = get_dataset_pipeline(dev_source_sentences)
    test_dataset = get_dataset_pipeline(test_source_sentences)
    ###

    # GECCC_DATASETS:
    def prepare_datasets(list_datasets):
        datasets = []
        refs = []
        eval_types = []
        for dataset in list_datasets:
            if not isinstance(dataset, str):
                dataset, eval_type = dataset
            else:
                eval_type = ['m2_scorer'] 
            source_sentences, gold_edits = load_annotation(dataset)
            if 'errant' in eval_type:
                ref_m2 = open(dataset).read().strip().split("\n\n")
                refs.append(ref_m2)
            else:
                refs.append(None)
            eval_types.append(eval_type)
            datasets.append((source_sentences, gold_edits, dataset))
        return datasets, refs, eval_types

    dev_geccc_datasets, _, _ = prepare_datasets(DEV_GECCC_DATASETS)
    test_geccc_datasets, _, _ = prepare_datasets(TEST_GECCC_DATASETS)
    retag_dev_geccc_datasets, _, _ = prepare_datasets(RETAG_DEV_GECCC_DATASETS)
    retag_test_geccc_datasets, _, _ = prepare_datasets(RETAG_TEST_GECCC_DATASETS)
    ###
    
    ### Prepare right model:
    if USE_F16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)

    with strategy.scope():
        if FROM_CONFIG:
            config = AutoConfig.from_pretrained(MODEL)
            model = TFAutoModelForSeq2SeqLM.from_config(config)
        else:
            model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

    if USE_F16:
        model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
        model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)
    ###

    # prepare udpipe tokenizer
    udpipe_tokenizer = UDPipeTokenizer("cs")
    
    def generate_and_score(unevaluated_checkpoint, dataset, predictions_file) -> float:
        step = int(unevaluated_checkpoint[5:])
        predictions_filepath = os.path.join(MODEL_CHECKPOINT_PATH, str(step) + "-" + predictions_file)

        ### Load model weights for evaluation
        model.load_weights(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint + "/")).expect_partial()
        ###

        print(f"Eval: {unevaluated_checkpoint}")

        print("Generating...")
        predicted_sentences = []
        for i, batch in enumerate(dataset):
            preds = model.generate(
                batch['input_ids'], 
                max_length=MAX_EVAL_LENGTH,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                )
            batch_sentences = tokenizer.batch_decode(preds, skip_special_tokens=True)
            predicted_sentences = predicted_sentences + batch_sentences
        print("End of generating...")

        print("Udpipe tokenization...")
        tokenized_predicted_sentences = []
        for i, line in enumerate(predicted_sentences):
            tokenization = udpipe_tokenizer.tokenize(line)
            sentence = " ".join([token.string for tokens_of_part in tokenization for token in tokens_of_part]) if len(tokenization) > 0 else ""
            tokenized_predicted_sentences.append(sentence)
        print("End of tokenization...")

        print("Write predictions...")
        with open(predictions_filepath, "w") as file:
            for sentence in tokenized_predicted_sentences:
                file.write(sentence + "\n")
        print("End of writing predictions...")
        return

    while True:
        if os.path.isdir(MODEL_CHECKPOINT_PATH):
            unevaluated = [f for f in os.listdir(MODEL_CHECKPOINT_PATH) if f.startswith('ckpt')]
            numbers = np.array([int(u[5:]) for u in unevaluated])
            numbers = sorted(numbers)
            unevaluated = ["ckpt-" + str(number) for number in numbers]
            print(unevaluated)

            if len(unevaluated) == 0:
                time.sleep(10)
                continue
            
            for unevaluated_checkpoint in unevaluated:
                try:
                    generate_and_score(unevaluated_checkpoint, dev_dataset, FILE_DEV_PREDICTIONS)
                    generate_and_score(unevaluated_checkpoint, test_dataset, FILE_TEST_PREDICTIONS)  
                    
                    ### GECCC:
                    def eval_splitted_dataset(datasets, unevaluated_checkpoint):
                        if len(datasets) == 0:
                            return
                        for i, dataset_zip in enumerate(datasets):
                            source_sentences, gold_edits, dataset_path = dataset_zip
                            dataset = get_dataset_pipeline(source_sentences)
                            file_predictions = os.path.splitext(os.path.basename(dataset_path))[0] + "_prediction.txt"
                            generate_and_score(unevaluated_checkpoint, dataset, file_predictions)

                    evaluate_every_two = False

                    if evaluate_every_two or (int(unevaluated_checkpoint[5:]) % EVAL_GECCC_EVERY == 0):
                        eval_splitted_dataset(dev_geccc_datasets, unevaluated_checkpoint)
                        eval_splitted_dataset(test_geccc_datasets, unevaluated_checkpoint)

                        eval_splitted_dataset(retag_dev_geccc_datasets, unevaluated_checkpoint)
                        eval_splitted_dataset(retag_test_geccc_datasets, unevaluated_checkpoint)

                        for i, dataset_zip in enumerate(datasets):
                            source_sentences, gold_edits, dataset_path = dataset_zip
                            dataset = get_dataset_pipeline(source_sentences)
                            file_predictions = os.path.splitext(os.path.basename(dataset_path))[0] + "_prediction.txt"
                            generate_and_score(unevaluated_checkpoint, dataset, file_predictions)


                    print(f"Delete: {os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint)}")
                    os.rename(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint), os.path.join(MODEL_CHECKPOINT_PATH, 'saved-' + unevaluated_checkpoint))

                except Exception as e:
                    print(e)
                    print("Something went wrong... Try again...")

        time.sleep(10)
