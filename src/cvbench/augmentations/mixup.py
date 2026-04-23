"""
Mixup augmentation — batch-level blending of signal images with a background class.

Only pairs where at least one image is the background class are blended.
Signal+signal pairs are left unchanged, preserving clean class boundaries.

To remove this feature entirely:
  - delete this file
  - remove the mixup block in services/training.py
  - remove --mixup-alpha and --mixup-background-class from cli/train.py
"""
import tensorflow as tf


def build_mixup_fn(background_class_idx: int, alpha: float = 0.2):
    """
    Returns a tf.data batch-map function that blends signal images with
    background-class images sampled from the same batch.

    Args:
        background_class_idx: Integer index of the background/negative class.
        alpha: Beta distribution shape parameter. 0.2 is standard — bimodal,
               so most blends are strongly one class or the other.

    Returns:
        A @tf.function suitable for dataset.map().
    """
    bg_idx = tf.constant(background_class_idx, dtype=tf.int64)
    _alpha = float(alpha)

    def _sample_beta(n):
        # Beta(a, a) via ratio of Gamma samples — avoids tfp dependency
        g1 = tf.random.gamma([n], alpha=_alpha)
        g2 = tf.random.gamma([n], alpha=_alpha)
        return g1 / (g1 + g2 + 1e-8)

    @tf.function
    def mixup_fn(x, y):
        batch_size = tf.shape(x)[0]

        # create mixing partners by random shuffle
        perm = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, perm)
        y2 = tf.gather(y, perm)

        # identify which items are background in each pair
        labels1 = tf.cast(tf.argmax(y,  axis=1), tf.int64)
        labels2 = tf.cast(tf.argmax(y2, axis=1), tf.int64)
        should_mix = tf.logical_or(
            tf.equal(labels1, bg_idx),
            tf.equal(labels2, bg_idx),
        )

        # per-image lambda from Beta distribution
        lam = _sample_beta(batch_size)
        lam_x = tf.reshape(lam, [-1, 1, 1, 1])
        lam_y = tf.reshape(lam, [-1, 1])

        x_mix = lam_x * x + (1.0 - lam_x) * x2
        y_mix = lam_y * y + (1.0 - lam_y) * y2

        # gate: only apply blend where at least one partner is background
        gate = tf.cast(should_mix, tf.float32)
        gate_x = tf.reshape(gate, [-1, 1, 1, 1])
        gate_y = tf.reshape(gate, [-1, 1])

        x_out = gate_x * x_mix + (1.0 - gate_x) * x
        y_out = gate_y * y_mix + (1.0 - gate_y) * y

        return x_out, y_out

    return mixup_fn
