import tensorflow as tf

class MaskedSparseCategoricalCrossEntropy(tf.keras.losses.Loss):
    # source: https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/modeling_tf_utils.py#L210
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name=None):
        super().__init__(reduction, name)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)

    def call(self, y_true, y_pred):
        return self.hf_compute_loss(y_true, y_pred)

    def hf_compute_loss(self, labels, logits):
        unmasked_loss = self.loss_func(tf.nn.relu(labels), logits)
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return reduced_masked_loss