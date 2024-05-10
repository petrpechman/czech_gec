import os
import json

import tensorflow as tf

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM

from tensorflow.keras import mixed_precision

from src.utils import dataset_utils
from src.utils import introduce_errors
from src.utils import create_errors

from src.utils.components.callbacks import MyBackupAndRestore
from src.utils.components.losses import MaskedSparseCategoricalCrossEntropy

from multiprocessing import Process, Manager


def main(config_filename: str):

    with open(config_filename) as json_file:
        config = json.load(json_file)

    with open(config['errors_config']) as json_file:
        errors_config = json.load(json_file)

    SEED = config['seed']

    # data loading
    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']
    MAX_LENGTH = config['max_length']
    SHUFFLE_BUFFER = config['shuffle_buffer']
    BUCKET_BOUNDARIES = config['bucket_boundaries']
    BUCKET_BATCH_SIZES_PER_GPU = config['bucket_batch_sizes_per_gpu']
    # data from file
    ERRORS_FROM_FILE = config.get('errors_from_file', False)
    REVERTED_PIPELINE = config.get('reverted_pipeline', False)

    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config'] # means from scratch
    STEPS_PER_EPOCH = config['steps_per_epoch']
    EPOCHS = config['epochs']
    USE_F16 = config['use_f16']

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']
    LR = OPTIMIZER_PARAMS.get('learning_rate', None)

    # loss
    LOSS = config['loss']

    # GEL config
    LANG = config['lang']
    TOKEN_FILE = config['token_file']
    TOKEN_ERR_DISTRIBUTION = config['token_err_distribution']
    DERINET_DIST = config['derinet_dist']
    CHAR_ERR_DISTRIBUTION = config['char_err_distribution']
    TOKEN_ERR_PROB = config['token_err_prob']   
    CHAR_ERR_PROB = config['char_err_prob']

    # logs
    LOG_FILE = config['log_file']
    PROFILE_BATCH = config.get('profile_batch', None)
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    BACKUP_DIR =  config['backup_dir']
    COUNT_OUTPUT = config.get('count_output', None)

    # mixture of datasets:
    MIXTURE_DATASET_PATHS = config.get('mixture_dataset_paths', None)
    RATIO_MIX = config.get('ratio_mix', [2, 1]) # first is main pipeline

    # input edits
    LABEL_PAD_VALUE = -100
    MODEL_TYPE = ""
    if MODEL in ["google/mt5-small", "google/mt5-base", "google/mt5-large"]:
        MODEL_TYPE = "T5"
    else:
        MODEL_TYPE = "Bart-mine"
    print(MODEL_TYPE)

    ### Init 
    tf.random.set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokens = introduce_errors.get_token_vocabulary(TOKEN_FILE)
    characters = introduce_errors.get_char_vocabulary(LANG)
    strategy = tf.distribute.MirroredStrategy()
    num_div = strategy.num_replicas_in_sync
    print('Number of devices: %d' % num_div)
    bucket_batch_sizes = [bucket_batch_size * num_div for bucket_batch_size in BUCKET_BATCH_SIZES_PER_GPU]
    print("Bucket batch size: ", bucket_batch_sizes)
    if REVERTED_PIPELINE:
        print('It is used REVERTED_PIPELINE.')
    ###

    ### Dataset loading:
    manager = Manager()
    queue = manager.Queue(4 * NUM_PARALLEL)
    if not ERRORS_FROM_FILE:
        error_generator = create_errors.ErrorGenerator(
            errors_config, tokens, characters, 
            CHAR_ERR_DISTRIBUTION, CHAR_ERR_PROB, 0.01,
            TOKEN_ERR_DISTRIBUTION, TOKEN_ERR_PROB, 0.2,
            DERINET_DIST)
        gel = None
        # gel = load_data.GenereteErrorLine(
        #     tokens, characters, LANG, 
        #     TOKEN_ERR_DISTRIBUTION, CHAR_ERR_DISTRIBUTION, 
        #     TOKEN_ERR_PROB, CHAR_ERR_PROB)
    else:
        gel = None
        error_generator = None

    # main process that creates pool, goes over possible files and manage other read processes
    process = Process(
                target=load_data.data_generator, 
                args=(queue, DATA_PATHS, NUM_PARALLEL, gel, tokenizer, MAX_LENGTH, 
                      ERRORS_FROM_FILE, REVERTED_PIPELINE, error_generator, LANG, 
                      COUNT_OUTPUT, ))

    process.start()

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(queue.get, None),
        output_types={
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "tokenized_target_line": tf.int32,
                    "original_sentence": tf.string,
                    "correct_sentence": tf.string,
                },
        output_shapes={
                    "input_ids": (None, ),
                    "attention_mask": (None, ),
                    "tokenized_target_line": (None, ),
                    "original_sentence": (),
                    "correct_sentence": (),
                })

    dataset = dataset.map(lambda input_batch: dataset_utils.fix_format(input_batch, MODEL_TYPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if MIXTURE_DATASET_PATHS:
        ### Dataset akces:
        num_parallel_akces = 2
        manager_akces = Manager()
        queue_akces = manager_akces.Queue(4 * num_parallel_akces)
        gel_akces = None
        error_generator_akces = None

        process_akces = Process(
                    target=load_data.data_generator, 
                    args=(queue_akces, MIXTURE_DATASET_PATHS, num_parallel_akces, 
                          gel_akces, tokenizer, MAX_LENGTH, 
                          True, REVERTED_PIPELINE, error_generator_akces, LANG, 
                          COUNT_OUTPUT, ))
        process_akces.start()
        dataset_akces = tf.data.Dataset.from_generator(
            lambda: iter(queue_akces.get, None),
            output_types={
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "tokenized_target_line": tf.int32,
                        "original_sentence": tf.string,
                        "correct_sentence": tf.string,
                    },
            output_shapes={
                        "input_ids": (None, ),
                        "attention_mask": (None, ),
                        "tokenized_target_line": (None, ),
                        "original_sentence": (),
                        "correct_sentence": (),
                    })
        dataset_akces = dataset_akces.map(lambda input_batch: dataset_utils.fix_format(input_batch, MODEL_TYPE), 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_akces = dataset_akces.map(dataset_utils.split_features_and_labels, 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ### Mixture:
        r1 = RATIO_MIX[0] # 2
        r2 = RATIO_MIX[1] # 1
        b1 = dataset.ragged_batch(r1)
        b2 = dataset_akces.ragged_batch(r2)
        zipped = tf.data.Dataset.zip((b1, b2)).map(dataset_utils.merge_ragged_batches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        zipped = zipped.unbatch() # lze mozna nahradit s .rebatch(1)
        zipped = zipped.batch(1)
        zipped = zipped.map(dataset_utils.retype, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = zipped.unbatch()
        ###

    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda x, y: tf.shape(x['input_ids'])[0], # zde asi chyba
            bucket_boundaries=BUCKET_BOUNDARIES,
            bucket_batch_sizes=bucket_batch_sizes
    )
    dataset = dataset.map(lambda x, y: dataset_utils.change_value(x, y, 0, LABEL_PAD_VALUE, MODEL_TYPE))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if USE_F16:
        if MODEL_TYPE == "T5":
            policy = mixed_precision.Policy('mixed_bfloat16')
        else:
            policy = mixed_precision.Policy('mixed_float16')    
        # policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    with strategy.scope():
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
            learning_rate = LRSchedule(OPTIMIZER_PARAMS['warmup_steps'], MAX_LENGTH)
            del OPTIMIZER_PARAMS['warmup_steps']
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'CosineDecay':
            cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(**OPTIMIZER_PARAMS)
            optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate=cosine_decay_scheduler)
        ###

        ### Loss:
        loss = None   
        if LOSS == "SCC":
            loss = MaskedSparseCategoricalCrossEntropy()
        ###

        ### Model
        if FROM_CONFIG:
            # means from scratch
            config = AutoConfig.from_pretrained(MODEL)
            model = TFAutoModelForSeq2SeqLM.from_config(config)
        else:
            print("Use pretrained model...")
            model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)
        ###

    ### Callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINT_PATH, 'ckpt-{epoch}/'),
        save_weights_only=True,
        save_freq="epoch")
    
    model_checkpoint_optimizer = MyBackupAndRestore(
        os.path.join(MODEL_CHECKPOINT_PATH, "optimizer"), optimizer, model,
        epoch_name="opt-ckpt",
        max_to_keep=None,
    )

    mybackup = MyBackupAndRestore(BACKUP_DIR, optimizer, model, max_to_keep=1)
    status = mybackup.checkpoint.restore(mybackup.manager.latest_checkpoint)
    status_optimizer = model_checkpoint_optimizer.checkpoint.restore(mybackup.manager.latest_checkpoint)
    print("STATUS:", status)
    initial_epoch = mybackup._ckpt_saved_epoch
    print("INITIAL EPOCH:", int(initial_epoch))

    profiler = None
    if PROFILE_BATCH:
        profiler = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_FILE, 
            profile_batch=PROFILE_BATCH)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_FILE, 
        histogram_freq=1)

    callbacks = [
        model_checkpoint,
        mybackup,
        model_checkpoint_optimizer,
        profiler,
        tensorboard_callback
    ]

    callbacks = [callback for callback in callbacks if callback is not None]
    ###

    with strategy.scope():
        if USE_F16 and MODEL_TYPE == "Bart-mine":
            model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
            model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)

        if LR:
            optimizer.learning_rate = tf.Variable(LR)
            optimizer._learning_rate = tf.Variable(LR)

        print("LEARNING RATE:")
        print(optimizer.learning_rate)
        print(optimizer._learning_rate)
        print("--------------")

    try:
        ### Train
        if STEPS_PER_EPOCH:
            model.fit(
                dataset, 
                initial_epoch=int(initial_epoch),
                callbacks=callbacks, 
                epochs=EPOCHS, 
                steps_per_epoch=STEPS_PER_EPOCH)
        else:
            model.fit(
                dataset,
                initial_epoch=int(initial_epoch),
                callbacks=callbacks, 
                epochs=EPOCHS)
        ###
    finally:
        process_akces.join()