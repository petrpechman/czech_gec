import sys
sys.path.append('..')

import os
import time
import shutil
import errant
import numpy as np
import tensorflow as tf

from typing import Tuple

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

from m2scorer.levenshtein import batch_multi_pre_rec_f1_part
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import dataset_utils
from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer
from utils.retag import retag_edits

from collections import Counter
from errant.commands.compare_m2 import simplify_edits, process_edits, merge_dict
from errant.commands.compare_m2 import compareEdits, computeFScore
# from errant.commands.compare_m2 import evaluate_edits

import multiprocessing
from multiprocessing.pool import Pool


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def write_evals(file_writer, 
                m2scorer_tp, m2scorer_fp, m2scorer_fn, 
                errant_tp, errant_fp, errant_fn,
                best_cats, step, beta, eval_type):
    if 'm2_scorer' in eval_type:
        m2scorer_p  = (1.0 * m2scorer_tp) / (m2scorer_tp + m2scorer_fp) if (m2scorer_tp + m2scorer_fp) > 0 else 0
        m2scorer_r  = (1.0 * m2scorer_tp) / (m2scorer_tp + m2scorer_fn) if (m2scorer_tp + m2scorer_fn) > 0 else 0
        m2scorer_f_score = (1.0+beta*beta) * m2scorer_p * m2scorer_r / (beta*beta*m2scorer_p+m2scorer_r) if (m2scorer_p+m2scorer_r) > 0 else 0

        print("Write into files...")
        with file_writer.as_default():
            tf.summary.scalar('epoch_m2scorer_precision', m2scorer_p, step)
            tf.summary.scalar('epoch_m2scorer_recall', m2scorer_r, step)
            tf.summary.scalar('epoch_m2scorer_f_score', m2scorer_f_score, step)
        print("End of writing into files...")

    if 'errant' in eval_type:
        errant_p  = (1.0 * errant_tp) / (errant_tp + errant_fp) if (errant_tp + errant_fp) > 0 else 0
        errant_r  = (1.0 * errant_tp) / (errant_tp + errant_fn)  if (errant_tp + errant_fn) > 0 else 0
        errant_f_score = (1.0+beta*beta) * errant_p * errant_r / (beta*beta*errant_p+errant_r) if (errant_p+errant_r) > 0 else 0

        print("Write into files...")
        with file_writer.as_default():
            tf.summary.scalar('epoch_errant_precision', errant_p, step)
            tf.summary.scalar('epoch_errant_recall', errant_r, step)
            tf.summary.scalar('epoch_errant_f_score', errant_f_score, step)
        print("End of writing into files...")

        print("Write specific errors...")
        with file_writer.as_default():
            text_lines = []
            for k, v in best_cats.items():
                tp = v[0]
                fp = v[1]
                fn = v[2]
                text_lines.append(k + ": " + f"tp: {tp}," + f" fp: {fp}," + f" fn: {fn}" + "\n\n")
                p  = (1.0 * tp) / (tp + fp) if (tp + fp) > 0 else 0
                r  = (1.0 * tp) / (tp + fn)  if (tp + fn) > 0 else 0
                f = (1.0+beta*beta) * p * r / (beta*beta*p+r) if (p+r) > 0 else 0
                description = f"gold_p: {tp+fn}, tp: {tp}, fp: {fp}, fn: {fn}"
                tf.summary.scalar(f"precision_spec_err_{k}", p, step, description=description)
                tf.summary.scalar(f"recall_spec_err_{k}", r, step, description=description)
                tf.summary.scalar(f"f_score_spec_err_{k}", f, step, description=description)
            text = "".join(text_lines)
            tf.summary.text("errors", text, step)
        print("End of writing specific errors...")

def noop_edit(id: int = 0):
    result = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(id)
    return result

def create_m2(annotator, source_sentence, predicted_sentence):
    orig = source_sentence
    cor = predicted_sentence
    cor_id = 0

    lev = False
    merge = "all-split"

    orig = annotator.parse(orig)
    output = " ".join(["S"] + [token.text for token in orig]) + "\n"

    cor = cor.strip()
    if orig.text.strip() == cor:
        output = output + noop_edit(cor_id) + "\n"
    else:
        cor = annotator.parse(cor)
        edits = annotator.annotate(orig, cor, lev, merge)
        for edit in edits:
            output = output + edit.to_m2(cor_id) + "\n"
    
    return output.strip()

def retag(m2_sentence: str) -> str:
    m2_lines = m2_sentence.split('\n')
    edits = retag_edits(m2_lines)
    m2_edits = [edit.to_m2() for edit in edits]
    m2_edits.insert(0, m2_lines[0])
    m2_sentence = "\n".join(m2_edits)
    return m2_sentence

def init_worker(max_unchanged_words_p, beta_p, ignore_whitespace_casing_p, verbose_p, very_verbose_p):
    global max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose
    max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose = max_unchanged_words_p, beta_p, ignore_whitespace_casing_p, verbose_p, very_verbose_p

def wrapper_func_m2scorer(tuple_items) -> Tuple[int, int, int]:
    sentence, source_sentence, gold_edit = tuple_items
    sentence, source_sentence, gold_edit = [sentence], [source_sentence], [gold_edit]
    stat_correct, stat_proposed, stat_gold = batch_multi_pre_rec_f1_part(
        sentence, 
        source_sentence, 
        gold_edit,
        max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose)
    return stat_correct, stat_proposed, stat_gold

class Args:
    def __init__(self, beta) -> None:
        self.beta = beta
        self.dt = None
        self.ds = None
        self.single = None
        self.multi = None
        self.filt = None
        self.cse = None
        self.verbose = None

def evaluate_edits(hyp_dict, ref_dict, args, best):
    best_tp, best_fp, best_fn, best_f = 0, 0, 0, -1
    best_cat = {}
    for hyp_id in hyp_dict.keys():
        for ref_id in ref_dict.keys():
            tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
            _, _, f = computeFScore(tp+best["tp"], fp+best["fp"], fn+best["fn"], args.beta)
            if     (f > best_f) or \
                (f == best_f and tp > best_tp) or \
                (f == best_f and tp == best_tp and fp < best_fp) or \
                (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                best_tp, best_fp, best_fn = tp, fp, fn
                best_f = f
                best_cat = cat_dict
    local_best_dict = {"tp":best_tp, "fp":best_fp, "fn":best_fn}
    return local_best_dict, best_cat

def join_dicts(dict1: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] = dict1[key] + dict2[key]
    return dict1

def init_worker_errant(beta_p):
    global beta, global_errant_tp, global_errant_fp, global_errant_fn
    beta = beta_p
    global_errant_tp = multiprocessing.Value('i', 0)
    global_errant_fp = multiprocessing.Value('i', 0)
    global_errant_fn = multiprocessing.Value('i', 0)
    # global_errant_best_dict = multiprocessing.Array('i', {"tp": 0, "fp": 0, "fn": 0})

def wrapper_func_errant(sent):
    args = Args(beta)
    hyp_edits = simplify_edits(sent[0])
    ref_edits = simplify_edits(sent[1])
    hyp_dict = process_edits(hyp_edits, args)
    ref_dict = process_edits(ref_edits, args)
    errant_best_dict = {"tp": 0, "fp": 0, "fn": 0}
    # with global_errant_best_dict.get_lock():
    with global_errant_tp.get_lock(), global_errant_fp.get_lock(), global_errant_fn.get_lock():
        errant_best_dict["tp"] = global_errant_tp.value
        errant_best_dict["fp"] = global_errant_fp.value
        errant_best_dict["fn"] = global_errant_fn.value
        # print("Loading: ", errant_best_dict)
    count_dict, cat_dict = evaluate_edits(hyp_dict, ref_dict, args, errant_best_dict)
    with global_errant_tp.get_lock(), global_errant_fp.get_lock(), global_errant_fn.get_lock():
        global_errant_tp.value = global_errant_tp.value + count_dict['tp']
        global_errant_fp.value = global_errant_fp.value + count_dict['fp']
        global_errant_fn.value = global_errant_fn.value + count_dict['fn']
        # global_errant_best_dict = join_dicts(global_errant_best_dict, count_dict)
        # print(f"Saving: TP:{global_errant_tp.value}, FP:{global_errant_fp.value}, FN:{global_errant_fn.value}")
    return count_dict, cat_dict

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

    def compute_metrics_m2scorer(tokenized_predicted_sentences, source_sentences, gold_edits):
        '''
        Goes through predicted sentences and computes true positives (stat_correct), 
        TP+FN (stat_gold), TP+FP (stat_proposed) for every batch.
        Finally it computes precision, recall and f score. 
        '''
        total_stat_correct, total_stat_proposed, total_stat_gold = 0, 0, 0

        # SORT:
        lenlist = [len(s) for s in tokenized_predicted_sentences]
        sortedindex = np.argsort(lenlist)[::-1]
        sort_tokenized_predicted_sentences = ['nothing'] * len(tokenized_predicted_sentences)
        sort_source_sentences = ['nothing'] * len(tokenized_predicted_sentences)
        sort_gold_edits = ['nothing'] * len(tokenized_predicted_sentences)
        for i in range(len(tokenized_predicted_sentences)):    
            sort_tokenized_predicted_sentences[i] = tokenized_predicted_sentences[sortedindex[i]]
            sort_source_sentences[i] = source_sentences[sortedindex[i]]
            sort_gold_edits[i] = gold_edits[sortedindex[i]]
        #

        with Pool(processes=NUM_EVAL_PROCESSES * 2, initializer=init_worker, initargs=(MAX_UNCHANGED_WORDS, BETA, IGNORE_WHITESPACE_CASING, VERBOSE, VERY_VERBOSE,)) as pool:
            result_iterator = pool.imap(
                wrapper_func_m2scorer, 
                zip(sort_tokenized_predicted_sentences, sort_source_sentences, sort_gold_edits)
            )
            pool.close()
            pool.join()

        for stat_correct, stat_proposed, stat_gold in result_iterator:
            total_stat_correct += stat_correct
            total_stat_proposed += stat_proposed
            total_stat_gold += stat_gold

        return total_stat_correct, total_stat_proposed, total_stat_gold
    
    def generate_and_score(unevaluated_checkpoint, dataset, source_sentences, gold_edits, output_dir, predictions_file,
                           ref_m2, eval_type) -> float:
        m2scorer_f_score  = 0
        m2scorer_tp, m2scorer_fp, m2scorer_fn = 0, 0, 0
        errant_tp, errant_fp, errant_fn = 0, 0, 0
        best_cats = None

        step = int(unevaluated_checkpoint[5:])
        result_dir = os.path.join(MODEL_CHECKPOINT_PATH, output_dir)
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

        file_writer = tf.summary.create_file_writer(result_dir)
        if 'm2_scorer' in eval_type:
            print("Compute metrics m2 scorer...")
            total_stat_correct, total_stat_proposed, total_stat_gold = compute_metrics_m2scorer(tokenized_predicted_sentences, source_sentences, gold_edits)
            m2scorer_tp = total_stat_correct
            m2scorer_fp = total_stat_proposed - m2scorer_tp
            m2scorer_fn = total_stat_gold - m2scorer_tp

            m2scorer_p  = total_stat_correct / total_stat_proposed if total_stat_proposed > 0 else 0
            m2scorer_r  = total_stat_correct / total_stat_gold if total_stat_gold > 0 else 0
            m2scorer_f_score = (1.0+BETA*BETA) * m2scorer_p * m2scorer_r / (BETA*BETA*m2scorer_p+m2scorer_r) if (m2scorer_p+m2scorer_r) > 0 else 0
            print("End of computing m2 scorer...")

            print("Write into files...")
            with file_writer.as_default():
                tf.summary.scalar('epoch_m2scorer_precision', m2scorer_p, step)
                tf.summary.scalar('epoch_m2scorer_recall', m2scorer_r, step)
                tf.summary.scalar('epoch_m2scorer_f_score', m2scorer_f_score, step)
            print("End of writing into files...")
        if 'errant' in eval_type:
            hyp_m2 = []
            annotator = errant.load('cs')
            for source_sentence, tokenized_predicted_sentence in zip(source_sentences, tokenized_predicted_sentences):
                m2_sentence = create_m2(annotator, source_sentence, tokenized_predicted_sentence)
                m2_sentence = retag(m2_sentence)
                hyp_m2.append(m2_sentence)

            # SORT:
            lenlist = [len(s) for s in tokenized_predicted_sentences]
            sortedindex = np.argsort(lenlist)[::-1]
            sort_hyp_m2 = ['nothing'] * len(tokenized_predicted_sentences)
            sort_ref_m2 = ['nothing'] * len(tokenized_predicted_sentences)
            for i in range(len(tokenized_predicted_sentences)):    
                sort_hyp_m2[i] = hyp_m2[sortedindex[i]]
                sort_ref_m2[i] = ref_m2[sortedindex[i]]
            #
                
            with Pool(processes=NUM_EVAL_PROCESSES * 2, initializer=init_worker_errant, initargs=(BETA,)) as pool:
                result_iterator = pool.imap(
                    wrapper_func_errant, 
                    zip(sort_hyp_m2, sort_ref_m2)
                )
                pool.close()
                pool.join()

            best_dict = Counter({"tp":0, "fp":0, "fn":0})
            best_cats = {}

            for count_dict, cat_dict in result_iterator:
                best_dict += Counter(count_dict)
                best_cats = merge_dict(best_cats, cat_dict)
            
            errant_tp = best_dict['tp']
            errant_fp = best_dict['fp']
            errant_fn = best_dict['fn']

            errant_p  = (1.0 * errant_tp) / (errant_tp + errant_fp) if (errant_tp + errant_fp) > 0 else 0
            errant_r  = (1.0 * errant_tp) / (errant_tp + errant_fn)  if (errant_tp + errant_fn) > 0 else 0
            errant_f_score = (1.0+BETA*BETA) * errant_p * errant_r / (BETA*BETA*errant_p+errant_r) if (errant_p+errant_r) > 0 else 0

            print("Write into files...")
            with file_writer.as_default():
                tf.summary.scalar('epoch_errant_precision', errant_p, step)
                tf.summary.scalar('epoch_errant_recall', errant_r, step)
                tf.summary.scalar('epoch_errant_f_score', errant_f_score, step)
            print("End of writing into files...")

            print("Write specific errors...")
            with file_writer.as_default():
                text_lines = []
                for k, v in best_cats.items():
                    tp = v[0]
                    fp = v[1]
                    fn = v[2]
                    text_lines.append(k + ": " + f"tp: {tp}," + f" fp: {fp}," + f" fn: {fn}" + "\n\n")
                    p  = (1.0 * tp) / (tp + fp) if (tp + fp) > 0 else 0
                    r  = (1.0 * tp) / (tp + fn)  if (tp + fn) > 0 else 0
                    f = (1.0+BETA*BETA) * p * r / (BETA*BETA*p+r) if (p+r) > 0 else 0
                    description = f"gold_p: {tp+fn}, tp: {tp}, fp: {fp}, fn: {fn}"
                    tf.summary.scalar(f"precision_spec_err_{k}", p, step, description=description)
                    tf.summary.scalar(f"recall_spec_err_{k}", r, step, description=description)
                    tf.summary.scalar(f"f_score_spec_err_{k}", f, step, description=description)
                text = "".join(text_lines)
                tf.summary.text("errors", text, step)
            print("End of writing specific errors...")

        print("Write predictions...")
        with file_writer.as_default():
            text = "  \n".join(tokenized_predicted_sentences[0:40])
            tf.summary.text("predictions", text, step)
        with open(predictions_filepath, "w") as file:
            for sentence in tokenized_predicted_sentences:
                file.write(sentence + "\n")
        print("End of writing predictions...")

        return m2scorer_f_score, m2scorer_tp, m2scorer_fp, m2scorer_fn, errant_tp, errant_fp, errant_fn, best_cats

    last_evaluated = 'ckpt-0'
    while True:
        if os.path.isdir(MODEL_CHECKPOINT_PATH):
            unevaluated = [f for f in os.listdir(MODEL_CHECKPOINT_PATH) if f.startswith('ckpt')]
            numbers = np.array([int(u[5:]) for u in unevaluated])
            numbers = sorted(numbers)
            unevaluated = ["ckpt-" + str(number) for number in numbers]

            if len(unevaluated) == 0:
                time.sleep(10)
                continue

            last = unevaluated[-1]
            if last_evaluated == last:
                unevaluated = unevaluated[:-1] # Remove last
            elif last_evaluated != "ckpt-0":
                print(f"Rename: {os.path.join(MODEL_CHECKPOINT_PATH, last_evaluated)}")
                # shutil.rmtree(os.path.join(MODEL_CHECKPOINT_PATH, last_evaluated))
                os.rename(os.path.join(MODEL_CHECKPOINT_PATH, last_evaluated), os.path.join(MODEL_CHECKPOINT_PATH, 'saved-' + last_evaluated))
                # if int(last_evaluated[5:]) % 5 != 0:
                #     # Delete model with optimizer:
                #     opt_dir = os.path.join(MODEL_CHECKPOINT_PATH, "optimizer")
                #     selected_files = []
                #     for f in os.listdir(opt_dir):
                #         if f.startswith(last_evaluated):
                #             selected_files.append(f)
                #     for selected_file in selected_files:
                #         os.remove(os.path.join(opt_dir, selected_file))

            if BEST_CKPT_NAME in unevaluated:
                unevaluated.remove(BEST_CKPT_NAME)
            
            for unevaluated_checkpoint in unevaluated:
                try:
                    fscore_dev, _, _, _, _, _, _, _ = generate_and_score(unevaluated_checkpoint, dev_dataset, dev_source_sentences, dev_gold_edits, OUTPUT_DIR_DEV,
                                                                         FILE_DEV_PREDICTIONS, dev_ref_m2, EVAL_TYPE_DEV)
                    fscore_test, _, _, _, _, _, _, _ = generate_and_score(unevaluated_checkpoint, test_dataset, test_source_sentences, test_gold_edits, OUTPUT_DIR_TEST,
                                                                          FILE_TEST_PREDICTIONS, test_ref_m2, EVAL_TYPE_TEST)  
                    
                    ### GECCC:
                    def eval_splitted_dataset(datasets, refs, eval_types, unevaluated_checkpoint, filename_results: str):
                        if len(datasets) == 0:
                            return
                        total_m2scorer_tp, total_m2scorer_fp, total_m2scorer_fn = 0, 0, 0
                        total_errant_tp, total_errant_fp, total_errant_fn = 0, 0, 0
                        total_best_cats = {}
                        for i, dataset_zip in enumerate(datasets):
                            source_sentences, gold_edits, dataset_path = dataset_zip
                            dataset = get_dataset_pipeline(source_sentences)
                            output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
                            file_predictions = os.path.splitext(os.path.basename(dataset_path))[0] + "_prediction.txt"
                            m2scorer_f_score, m2scorer_tp, m2scorer_fp, m2scorer_fn, errant_tp, errant_fp, errant_fn, best_cats = generate_and_score(
                                unevaluated_checkpoint, dataset, source_sentences, gold_edits, output_dir, file_predictions, refs[i], eval_types[i])

                            total_m2scorer_tp += m2scorer_tp
                            total_m2scorer_fp += m2scorer_fp
                            total_m2scorer_fn += m2scorer_fn

                            total_errant_tp += errant_tp
                            total_errant_fp += errant_fp
                            total_errant_fn += errant_fn

                            if best_cats is not None:
                                total_best_cats = merge_dict(total_best_cats, best_cats)

                        result_dir = os.path.join(MODEL_CHECKPOINT_PATH, filename_results)
                        file_writer = tf.summary.create_file_writer(result_dir)
                        step = int(unevaluated_checkpoint[5:])
                        write_evals(file_writer, 
                                    total_m2scorer_tp, total_m2scorer_fp, total_m2scorer_fn, 
                                    total_errant_tp, total_errant_fp, total_errant_fn, 
                                    total_best_cats, step, BETA, eval_types[i])
                    
                    evaluate_every_two = False
                    if FIRST_CHECKPOINT and (int(unevaluated_checkpoint[5:]) - 16) < FIRST_CHECKPOINT:
                        evaluate_every_two = True
                        # if int(unevaluated_checkpoint[5:]) % 1 == 0:
                        #     evaluate_every_two = True

                    if evaluate_every_two or (int(unevaluated_checkpoint[5:]) % EVAL_GECCC_EVERY == 0):
                        eval_splitted_dataset(dev_geccc_datasets, dev_geccc_refs, dev_geccc_eval_types, unevaluated_checkpoint, "dev_total_geccc")
                        eval_splitted_dataset(test_geccc_datasets, test_geccc_refs, test_geccc_eval_types, unevaluated_checkpoint, "test_total_geccc")

                        eval_splitted_dataset(retag_dev_geccc_datasets, retag_dev_geccc_refs, retag_dev_geccc_eval_types, unevaluated_checkpoint, "retag_dev_total_geccc")
                        eval_splitted_dataset(retag_test_geccc_datasets, retag_test_geccc_refs, retag_test_geccc_eval_types, unevaluated_checkpoint, "retag_test_total_geccc")

                        for i, dataset_zip in enumerate(datasets):
                            source_sentences, gold_edits, dataset_path = dataset_zip
                            dataset = get_dataset_pipeline(source_sentences)
                            output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
                            file_predictions = os.path.splitext(os.path.basename(dataset_path))[0] + "_prediction.txt"
                            m2scorer_f_score, _, _, _, _, _, _, _  = generate_and_score(
                                unevaluated_checkpoint, dataset, source_sentences, gold_edits, output_dir, file_predictions, refs[i], eval_types[i])
                    

                    if BEST_CKPT_FILENAME and fscore_dev > BEST_CKPT_FSCORE:
                        BEST_CKPT_NAME = unevaluated_checkpoint
                        BEST_CKPT_FSCORE = fscore_dev
                        
                        json_object = json.dumps({
                             "name": BEST_CKPT_NAME,
                             "fscore": BEST_CKPT_FSCORE
                        })

                        with open(BEST_CKPT_FILENAME, "w") as outfile:
                            outfile.write(json_object)
                    else:
                        if unevaluated_checkpoint == last:
                            last_evaluated = last
                        else:
                            print(f"Rename: {os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint)}")
                            # shutil.rmtree(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint))
                            os.rename(os.path.join(MODEL_CHECKPOINT_PATH, last_evaluated), os.path.join(MODEL_CHECKPOINT_PATH, 'saved-' + last_evaluated))
                            # Delete model with optimizer:
                            # if int(unevaluated_checkpoint[5:]) % 5 != 0:
                            #     opt_dir = os.path.join(MODEL_CHECKPOINT_PATH, "optimizer")
                            #     selected_files = []
                            #     for f in os.listdir(opt_dir):
                            #         if f.startswith(unevaluated_checkpoint):
                            #             selected_files.append(f)

                            #     for selected_file in selected_files:
                            #         os.remove(os.path.join(opt_dir, selected_file))
                except Exception as e:
                    print(e)
                    print("Something went wrong... Try again...")

        time.sleep(10)
