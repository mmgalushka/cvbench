"""Penalty-reduced focal loss (heatmap) + masked L1 (size/offset).

Standard CornerNet/CenterNet loss: a modified focal loss on the heatmap that
down-weights the penalty near (but not at) a true peak — see the `beta`
Gaussian-penalty term below — plus masked L1 regression on box size and
center offset, computed only at object-center cells (``mask`` from the
target's last channel).
"""
from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="cvbench")
class CenterNetLoss(keras.losses.Loss):
    """Loss for the (G, G, C+4) CenterNet-style prediction against a
    (G, G, C+5) target — see detection/data.py::encode_target for the layout.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 2.0,
        beta: float = 4.0,
        size_weight: float = 0.1,
        offset_weight: float = 1.0,
        epsilon: float = 1e-6,
        name: str = "centernet_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        C = self.num_classes
        eps = self.epsilon

        hm_true = y_true[..., :C]
        size_true = y_true[..., C:C + 2]
        offset_true = y_true[..., C + 2:C + 4]
        mask = y_true[..., C + 4:C + 5]

        hm_pred = keras.ops.clip(y_pred[..., :C], eps, 1.0 - eps)
        size_pred = y_pred[..., C:C + 2]
        offset_pred = y_pred[..., C + 2:C + 4]

        # Penalty-reduced focal loss on the heatmap.
        pos_mask = keras.ops.cast(keras.ops.equal(hm_true, 1.0), "float32")
        neg_mask = 1.0 - pos_mask
        neg_weights = keras.ops.power(1.0 - hm_true, self.beta)

        pos_loss = -pos_mask * keras.ops.power(1.0 - hm_pred, self.alpha) * keras.ops.log(hm_pred)
        neg_loss = (
            -neg_mask * neg_weights * keras.ops.power(hm_pred, self.alpha)
            * keras.ops.log(1.0 - hm_pred)
        )

        num_pos = keras.ops.sum(pos_mask)
        norm = keras.ops.maximum(num_pos, 1.0)
        heatmap_loss = keras.ops.sum(pos_loss + neg_loss) / norm

        # Masked L1 on size/offset — only at object-center cells.
        size_loss = keras.ops.sum(keras.ops.abs(size_true - size_pred) * mask) / norm
        offset_loss = keras.ops.sum(keras.ops.abs(offset_true - offset_pred) * mask) / norm

        return heatmap_loss + self.size_weight * size_loss + self.offset_weight * offset_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "beta": self.beta,
            "size_weight": self.size_weight,
            "offset_weight": self.offset_weight,
            "epsilon": self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
