"""수집 매트릭스 설정 → 에피소드 목록 (CARLA 비의존)."""
from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from driveloop.config import PROJECT_ROOT, load_config


@dataclass
class CaptureCamera:
    width: int = 1280
    height: int = 720
    fov: float = 90.0
    x: float = 0.5
    z: float = 1.7


@dataclass
class CollectionConfig:
    name: str
    maps: list[str]
    weathers: dict[str, dict[str, float]]
    times: dict[str, dict[str, float]]
    traffic: dict[str, int]
    output_root: str = "data/raw"
    seed: int = 0
    repeats: int = 1
    frames_per_episode: int = 150
    capture_interval: float = 0.5
    warmup_seconds: float = 3.0
    fixed_delta_seconds: float = 0.05
    label_radius: float = 80.0
    camera: CaptureCamera = field(default_factory=CaptureCamera)

    @property
    def output_dir(self) -> Path:
        root = Path(self.output_root)
        return (root if root.is_absolute() else PROJECT_ROOT / root) / self.name

    @property
    def capture_every_ticks(self) -> int:
        return max(1, round(self.capture_interval / self.fixed_delta_seconds))

    @property
    def warmup_ticks(self) -> int:
        return round(self.warmup_seconds / self.fixed_delta_seconds)


@dataclass(frozen=True)
class EpisodeSpec:
    map: str
    weather: str
    time: str
    traffic: str
    repeat: int
    seed: int

    @property
    def episode_id(self) -> str:
        return f"{self.map}_{self.weather}_{self.time}_{self.traffic}_r{self.repeat}"

    def tags(self) -> dict[str, Any]:
        return {"map": self.map, "weather": self.weather, "time": self.time,
                "traffic": self.traffic, "repeat": self.repeat}


def episode_seed(base_seed: int, episode_id: str) -> int:
    """에피소드 ID로 seed를 정한다 → 매트릭스에 조건을 추가해도 기존 에피소드는 그대로 재현된다."""
    return (zlib.crc32(episode_id.encode()) ^ base_seed) & 0x7FFFFFFF


def expand_matrix(cfg: CollectionConfig) -> list[EpisodeSpec]:
    """모든 조건 조합. 맵 로딩 횟수를 줄이도록 맵 순서대로 묶는다."""
    specs = []
    for m, w, t, tr, r in itertools.product(cfg.maps, cfg.weathers, cfg.times, cfg.traffic,
                                            range(cfg.repeats)):
        spec = EpisodeSpec(m, w, t, tr, r, seed=0)
        specs.append(EpisodeSpec(m, w, t, tr, r, episode_seed(cfg.seed, spec.episode_id)))
    return specs


def weather_params(cfg: CollectionConfig, spec: EpisodeSpec) -> dict[str, float]:
    return {**cfg.weathers[spec.weather], **cfg.times[spec.time]}


def load_collection_config(path: str | Path) -> CollectionConfig:
    return load_config(CollectionConfig, path)
