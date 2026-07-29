"""Dataset e collate para o treino de SLT."""

from skeltrans.training.data.dataset import LandmarkTextDataset
from skeltrans.training.data.collate import make_collate

__all__ = ["LandmarkTextDataset", "make_collate"]
