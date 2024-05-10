import tensorflow as tf
from transformers.tf_utils import shape_list

###
#
# There are two possible types of model: T5 and Bart from scratch (Bart-mine)
#   If T5 is chosen:
#   - fix_format - creates input_ids, attention_mask and labels
#   - change_value - changes labels and creates decoder_input_ids, it changes 0 (pad_value) by -100 for labels 
#                    and it shifts labels right and change value -100 back into 0  
#   for example:
#       > ids = tf.constant([[1, 2, 3, 4, 0, 0]])
#       > x = {}
#       > change_value(x, ids, 0 , -100, 'T5')
#         ({'decoder_input_ids': <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[0, 1, 2, 3, 4, 0]], dtype=int32)>}, 
#          <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[   1,    2,    3,    4, -100, -100]], dtype=int32)>)
#
#   If Bart-mine is chosen:
#   - fix_format - creates input_ids, attention_mask, labels and decoder_input_ids
#   - change_value - only change 0 (pad value) by -100 in labels
###

def fix_format(input_batch, model_type):
    if model_type == "T5":
        dato = {
                "input_ids": input_batch["input_ids"],
                "attention_mask": input_batch["attention_mask"],
                "labels": input_batch["tokenized_target_line"],
            }
    elif model_type == "Bart-mine":
        dato = {
                "input_ids": input_batch["input_ids"],
                "attention_mask": input_batch["attention_mask"],
                "labels": input_batch["tokenized_target_line"][1:],
                "decoder_input_ids": input_batch["tokenized_target_line"][:-1]
            }
    return dato

def split_features_and_labels(input_batch):
    features = {key: tensor for key, tensor in input_batch.items() if key in ['input_ids', 'attention_mask', 'decoder_input_ids']}
    labels = {key: tensor for key, tensor in input_batch.items() if key in ['labels']}
    if len(features) == 1:
        features = list(features.values())[0]
    if len(labels) == 1:
        labels = list(labels.values())[0]
    if isinstance(labels, dict) and len(labels) == 0:
        return features
    else:
        return features, labels

def change_value(x, y, original_value, new_value, model_type):
    condition = tf.not_equal(y, original_value)
    changed_y = tf.where(condition, y, new_value)
    if model_type == "T5":    
        x['decoder_input_ids'] = _shift_right_t5(changed_y)
        return x, changed_y
    elif model_type == "Bart-mine":
        return x, changed_y

def _shift_right_t5(input_ids):
    # taken from https://github.com/huggingface/transformers/blob/6da93f5580e109fad5f7b523cf2b6e8a5bafb623/src/transformers/models/t5/modeling_t5.py#L880
    decoder_start_token_id = 0 # 0 is decoder start token for T5 tokenizer
    pad_token_id = 0

    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
        shifted_input_ids,
    )
    # "Verify that `labels` has only positive values and -100"
    assert_gte0 = tf.debugging.assert_greater_equal(
        shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype)
    )
    # Make sure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    return shifted_input_ids

def merge_ragged_batches(dato_a, dato_b):
    x_a, y_a = dato_a
    x_b, y_b = dato_b
    x = dict()
    for key in x_a.keys():
        c = tf.concat([x_a[key], x_b[key]], axis=0)
        x[key] = c
    y = tf.concat([y_a, y_b], axis=0)
    return x, y

def retype(x, y):
    x_new = dict()
    for key in x.keys():
        x_new[key] = x[key].to_tensor()
    y = y.to_tensor()
    return x_new, y