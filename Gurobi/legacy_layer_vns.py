from __future__ import annotations


class LegacyLayerAugmentedMixin:
    def _run_layer_augmented_main(self) -> float:
        raise NotImplementedError("layer_augmented has been retired; use resource_time_alns")
