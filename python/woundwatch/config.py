"""Project settings. All file paths derive from this module's location."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    seed: int = 20261031
    classifier_backbone: str = "resnet18"
    classifier_size: tuple[int, int] = (224, 224)
    segment_size: tuple[int, int] = (256, 256)
    folds: int = 5
    classifier_epochs: int = 8
    segmentation_epochs: int = 25
    classifier_gpu_batch: int = 16
    classifier_cpu_batch: int = 16
    segmentation_gpu_batch: int = 4
    segmentation_cpu_batch: int = 1
    learning_rate: float = 1e-4
    ring_dilation_pixels: int = 12
    defer_half_width: float = 0.05
    blur_variance_minimum: float = 1e-4
    maximum_clipped_fraction: float = 0.05
    minimum_wound_area_fraction: float = 0.002
    maximum_wound_area_fraction: float = 0.80
    bootstrap_replicates: int = 2000
    onnx_parity_tolerance: float = 1e-3
    onnx_verification_images: int = 20
    @property
    def raw(self): return self.root / "data" / "raw"
    @property
    def surgwound(self): return self.raw / "surgwound"
    @property
    def manifest(self): return self.surgwound / "infection_risk.csv"
    @property
    def kaggle(self): return self.raw / "Kaggle" / "Wound_dataset copy"
    @property
    def medetec(self): return self.raw / "Medetec" / "medetec-dataset"
    @property
    def fuseg(self): return self.raw / "FUSeg" / "Foot Ulcer Segmentation Challenge"
    @property
    def fuseg_external(self): return self.raw / "FUSeg" / "Medetec_foot_ulcer_224" / "test"
    @property
    def models(self): return self.root / "models"
    @property
    def results(self): return self.root / "results"
    @property
    def processed(self): return self.root / "data" / "processed"
    def create_dirs(self):
        for p in (self.models, self.results, self.processed): p.mkdir(parents=True, exist_ok=True)

CFG = Config()
