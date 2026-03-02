"""
trainer/ - Training pipeline package
======================================
    from trainer import VSLDataset, build_dataloaders, compute_split_counts
    from trainer import Trainer
"""

from trainer.dataset    import VSLDataset, build_dataloaders, compute_split_counts
from trainer.compute_split_counts.train_loop import Trainer

__all__ = [
    'VSLDataset', 'build_dataloaders', 'compute_split_counts',
    'Trainer',
]