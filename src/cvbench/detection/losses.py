"""YOLO-style detection loss: objectness BCE (with an ignore mask) + box
regression + multi-label class BCE, one instance per detection scale.

The model's per-scale output is ``(B, G, G, A*(5+C))`` raw logits (no
activation applied in the graph — see ``model.py``); the target is
``(B, G, G, A*(6+C))`` — see ``data.py``'s module docstring for the extra
``ignore`` channel. Both are reshaped to ``(B, G, G, A, ·)`` here before the
per-term losses are computed, all from logits for numerical stability.
"""
from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="cvbench")
class YoloLoss(keras.losses.Loss):
    """Loss for one detection scale's ``(G, G, A*(5+C))`` prediction against
    its ``(G, G, A*(6+C))`` target — see ``detection/data.py::encode_targets``
    for the target channel layout.
    """

    def __init__(
        self,
        num_classes: int,
        num_anchors: int,
        obj_weight: float = 1.0,
        noobj_weight: float = 0.5,
        box_weight: float = 1.0,
        cls_weight: float = 1.0,
        name: str = "yolo_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight

    def call(self, y_true, y_pred):
        C = self.num_classes
        A = self.num_anchors
        pred_ch = 5 + C
        true_ch = 6 + C

        pred_shape = keras.ops.shape(y_pred)
        batch, grid = pred_shape[0], pred_shape[1]
        y_pred = keras.ops.reshape(y_pred, (batch, grid, grid, A, pred_ch))
        y_true = keras.ops.reshape(y_true, (batch, grid, grid, A, true_ch))

        tx_t, ty_t = y_true[..., 0], y_true[..., 1]
        tw_t, th_t = y_true[..., 2], y_true[..., 3]
        obj_t = y_true[..., 4]
        ignore_t = y_true[..., 5]
        cls_t = y_true[..., 6:]

        tx_p, ty_p = y_pred[..., 0], y_pred[..., 1]
        tw_p, th_p = y_pred[..., 2], y_pred[..., 3]
        obj_p = y_pred[..., 4]
        cls_p = y_pred[..., 5:]

        pos_mask = obj_t
        # A cell is a true negative only if it's neither an object center
        # nor an ignored near-miss anchor.
        neg_mask = (1.0 - obj_t) * (1.0 - ignore_t)
        num_pos = keras.ops.maximum(keras.ops.sum(pos_mask), 1.0)

        xy_loss = keras.ops.sum(
            (
                keras.ops.binary_crossentropy(tx_t, tx_p, from_logits=True)
                + keras.ops.binary_crossentropy(ty_t, ty_p, from_logits=True)
            ) * pos_mask
        ) / num_pos
        wh_loss = keras.ops.sum(
            (keras.ops.square(tw_t - tw_p) + keras.ops.square(th_t - th_p)) * pos_mask
        ) / num_pos
        box_loss = xy_loss + wh_loss

        obj_bce = keras.ops.binary_crossentropy(obj_t, obj_p, from_logits=True)
        obj_loss = (
            self.obj_weight * keras.ops.sum(obj_bce * pos_mask)
            + self.noobj_weight * keras.ops.sum(obj_bce * neg_mask)
        ) / num_pos

        cls_bce = keras.ops.binary_crossentropy(cls_t, cls_p, from_logits=True)
        cls_loss = keras.ops.sum(cls_bce * pos_mask[..., None]) / num_pos

        return self.box_weight * box_loss + obj_loss + self.cls_weight * cls_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_anchors": self.num_anchors,
            "obj_weight": self.obj_weight,
            "noobj_weight": self.noobj_weight,
            "box_weight": self.box_weight,
            "cls_weight": self.cls_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
