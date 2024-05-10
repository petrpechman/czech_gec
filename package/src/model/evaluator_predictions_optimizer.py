import os
import time
import json
import numpy as np
import tensorflow as tf

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM

from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from src.utils import dataset_utils
from src.utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer
from src.utils.components.callbacks import MyBackupAndRestore


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

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']
    LR = OPTIMIZER_PARAMS.get('learning_rate', None)

    # data loading
    EVAL_TYPE_DEV, EVAL_TYPE_TEST = ['m2_scorer'], ['m2_scorer']
    M2_DATA_DEV = config['m2_data_dev']
    if not isinstance(M2_DATA_DEV, str):
        M2_DATA_DEV, EVAL_TYPE_DEV = M2_DATA_DEV

    M2_DATA_TEST = config['m2_data_test']
    if not isinstance(M2_DATA_TEST, str):
        M2_DATA_TEST, EVAL_TYPE_TEST = M2_DATA_TEST
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
    # USE_F16 = config['use_f16']
    USE_F16 = False
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']

    # evaluation
    MAX_UNCHANGED_WORDS = config['max_unchanged_words']
    BETA = config['beta']
    IGNORE_WHITESPACE_CASING = config['ignore_whitespace_casing']
    VERBOSE = config['verbose']
    VERY_VERBOSE = config['very_verbose']
    
    MAX_EVAL_LENGTH = config['max_eval_length']

    # TIMEOUT = config['timeout'] # it cat be useful for geccc

    # OUTPUT_DIR = 'results' # "m2_data": "../../data/geccc/dev/sorted_sentence.m2",
    OUTPUT_DIR_DEV = 'results-dev' # "m2_data": "../../data/akces-gec/dev/dev.all.m2",
    OUTPUT_DIR_TEST = 'results-test' # "m2_data": "../../data/akces-gec/test/test.all.m2",
    FILE_DEV_PREDICTIONS = 'predictions_dev.txt'
    FILE_TEST_PREDICTIONS = 'predictions_test.txt'

    BEST_CKPT_FILENAME = config.get("best_ckpt_filename", None)
    if BEST_CKPT_FILENAME:
        with open(BEST_CKPT_FILENAME) as json_file:
            best_ckpt = json.load(json_file)
        BEST_CKPT_NAME = best_ckpt['name']
        BEST_CKPT_FSCORE = best_ckpt['fscore']

    NUM_EVAL_PROCESSES = config.get('num_eval_processes', 4)
    EVAL_GECCC_EVERY = config.get('eval_geccc_every', 10)

    FIRST_CHECKPOINT = config.get('first_checkpoint', None)

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
    
    dev_source_sentences, dev_gold_edits = load_annotation(M2_DATA_DEV)
    test_source_sentences, test_gold_edits = load_annotation(M2_DATA_TEST)

    dev_ref_m2, test_ref_m2 = None, None
    if 'errant' in EVAL_TYPE_DEV:
        dev_ref_m2 = open(M2_DATA_DEV).read().strip().split("\n\n")
    if 'errant' in EVAL_TYPE_TEST:
        test_ref_m2 = open(M2_DATA_TEST).read().strip().split("\n\n")
    
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

    dev_geccc_datasets, dev_geccc_refs, dev_geccc_eval_types = prepare_datasets(DEV_GECCC_DATASETS)
    test_geccc_datasets, test_geccc_refs, test_geccc_eval_types = prepare_datasets(TEST_GECCC_DATASETS)
    retag_dev_geccc_datasets, retag_dev_geccc_refs, retag_dev_geccc_eval_types = prepare_datasets(RETAG_DEV_GECCC_DATASETS)
    retag_test_geccc_datasets, retag_test_geccc_refs, retag_test_geccc_eval_types = prepare_datasets(RETAG_TEST_GECCC_DATASETS)
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

        ### Optimizer:
        if OPTIMIZER_NAME == 'Adam':
            optimizer = tf.keras.optimizers.Adam(**OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'AdamW':
            optimizer = tf.keras.optimizers.experimental.AdamW(**OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'Adafactor':
            optimizer = tf.keras.optimizers.experimental.Adafactor(**OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'AdaptiveAdam':
            class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup_steps, d_model):
                    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                    self.d_model = tf.cast(d_model, tf.float32)

                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    lr = (1.0/tf.math.sqrt(self.d_model)) * tf.math.minimum(1.0 / tf.math.sqrt(step), (1.0 / tf.math.sqrt(self.warmup_steps)) * ((1.0 * step) / self.warmup_steps))
                    return lr
            learning_rate = LRSchedule(OPTIMIZER_PARAMS['warmup_steps'], MAX_EVAL_LENGTH)
            del OPTIMIZER_PARAMS['warmup_steps']
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'CosineDecay':
            cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(**OPTIMIZER_PARAMS)
            optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate=cosine_decay_scheduler)
        ###


        ### Load model weights for evaluation
        # model.load_weights(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint + "/")).expect_partial()

        mybackup = MyBackupAndRestore(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint), optimizer, model, max_to_keep=1)
        status = mybackup.checkpoint.restore(mybackup.manager.latest_checkpoint)
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

    last_evaluated = 'ckpt-0'
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
                    # if FIRST_CHECKPOINT and (int(unevaluated_checkpoint[5:]) - 10) < FIRST_CHECKPOINT:
                    #     evaluate_every_two = True
                        # if int(unevaluated_checkpoint[5:]) % 1 == 0:
                        #     evaluate_every_two = True

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
                    # shutil.rmtree(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint))
                    os.rename(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint), os.path.join(MODEL_CHECKPOINT_PATH, 'saved-' + unevaluated_checkpoint))

                except Exception as e:
                    print(e)
                    print("Something went wrong... Try again...")

        time.sleep(10)
