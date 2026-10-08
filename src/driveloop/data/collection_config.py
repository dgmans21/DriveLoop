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
class TrafficLightTiming:
    """신호등 단계별 시간(초). 노란불을 늘려 희소한 노란불 장면을 더 모으는 데 쓴다 (겉모습은 동일)."""
    green: float = 10.0
    yellow: float = 3.0
    red: float = 2.0


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
    repeat_start: int = 0                  # 새 seed로 추가 수집할 때 (v1이 r0이면 v2는 1부터)
    frames_per_episode: int = 150
    capture_interval: float = 0.5
    warmup_seconds: float = 3.0
    fixed_delta_seconds: float = 0.05
    label_radius: float = 80.0
    camera: CaptureCamera = field(default_factory=CaptureCamera)
    traffic_light_timing: TrafficLightTiming | None = None   # None이면 맵 기본값 유지
    yellow_capture_interval: float | None = None             # 노란불이 보이면 이 간격으로 저장 (None=끔)
    yellow_max_facing_angle: float = 70.0                    # '보인다' 판정 = 라벨러와 같은 기준
    walkers: dict[str, int] = field(default_factory=dict)    # 교통량 단계별 보행자 수 (v4~). 없으면 0명 (v1~v3)
    walker_cross_factor: float = 0.3                         # 보행자가 차도를 건너는 비율 (CARLA 기본 0)
    walker_running: float = 0.1                              # 뛰는 보행자 비율

    def walker_count(self, traffic: str) -> int:
        return self.walkers.get(traffic, 0)

    @property
    def output_dir(self) -> Path:
        root = Path(self.output_root)
        return (root if root.is_absolute() else PROJECT_ROOT / root) / self.name

    @property
    def capture_every_ticks(self) -> int:
        return max(1, round(self.capture_interval / self.fixed_delta_seconds))

    @property
    def yellow_capture_ticks(self) -> int | None:
        if self.yellow_capture_interval is None:
            return None
        return max(1, round(self.yellow_capture_interval / self.fixed_delta_seconds))

    @property
    def episode_ticks(self) -> int:
        """에피소드 길이(워밍업 제외). 장수가 아니라 시간 기준 → 노란불 추가 저장이 일반 장면을 줄이지 않는다."""
        return self.frames_per_episode * self.capture_every_ticks

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
                                            range(cfg.repeat_start, cfg.repeat_start + cfg.repeats)):
        spec = EpisodeSpec(m, w, t, tr, r, seed=0)
        specs.append(EpisodeSpec(m, w, t, tr, r, episode_seed(cfg.seed, spec.episode_id)))
    return specs


def weather_params(cfg: CollectionConfig, spec: EpisodeSpec) -> dict[str, float]:
    return {**cfg.weathers[spec.weather], **cfg.times[spec.time]}


def load_collection_config(path: str | Path) -> CollectionConfig:
    return load_config(CollectionConfig, path)
