"""에피소드 단위 원본 데이터 저장 (CARLA 비의존).

레이아웃:
  <episode_id>/
    episode.json     조건 태그, 날씨 파라미터, 카메라 내부 파라미터, 신호등 카탈로그, 통계
    frames.jsonl     프레임별 정답 (한 줄 = 한 프레임)
    rgb/000000.jpg   전방 RGB
    seg/000000.png   인스턴스 세그멘테이션 (R=semantic tag, G·B=instance id, 무손실)

`.partial` 디렉터리에 쓰다가 완료 시 원자적으로 이름을 바꾼다.
→ 중간에 죽어도 불완전한 에피소드가 완성본으로 섞이지 않고, 재실행하면 완성본은 건너뛴다.
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

EPISODE_FILE = "episode.json"
FRAMES_FILE = "frames.jsonl"


def is_complete(episode_dir: Path) -> bool:
    meta = episode_dir / EPISODE_FILE
    if not meta.exists():
        return False
    return json.loads(meta.read_text(encoding="utf-8")).get("status") == "complete"


class EpisodeWriter:
    def __init__(self, episode_dir: Path, jpeg_quality: int = 92, workers: int = 4) -> None:
        self.final_dir = episode_dir
        self.tmp_dir = episode_dir.with_name(episode_dir.name + ".partial")
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        (self.tmp_dir / "rgb").mkdir(parents=True)
        (self.tmp_dir / "seg").mkdir()
        self._jpeg_quality = jpeg_quality
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._futures: list[Future] = []
        self._frames = open(self.tmp_dir / FRAMES_FILE, "w", encoding="utf-8")
        self.count = 0

    def add(self, rgb: np.ndarray, seg: np.ndarray, meta: dict) -> None:
        """rgb/seg: (H, W, 3) uint8. 배열은 복사해서 백그라운드 스레드가 저장한다."""
        name = f"{self.count:06d}"
        rgb_rel, seg_rel = f"rgb/{name}.jpg", f"seg/{name}.png"
        rgb_img = Image.fromarray(np.ascontiguousarray(rgb))
        seg_img = Image.fromarray(np.ascontiguousarray(seg))
        self._futures.append(self._pool.submit(
            rgb_img.save, self.tmp_dir / rgb_rel, quality=self._jpeg_quality))
        self._futures.append(self._pool.submit(seg_img.save, self.tmp_dir / seg_rel))
        record = {"index": self.count, "rgb": rgb_rel, "seg": seg_rel, **meta}
        self._frames.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1

    def finalize(self, episode_meta: dict) -> Path:
        self._pool.shutdown(wait=True)
        for f in self._futures:
            f.result()  # 저장 실패 시 여기서 예외
        self._frames.close()
        meta = {**episode_meta, "num_frames": self.count, "status": "complete"}
        (self.tmp_dir / EPISODE_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        os.replace(self.tmp_dir, self.final_dir)
        return self.final_dir

    def abort(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._frames.close()
        # .partial은 남겨 둔다 (원인 확인용). 다음 실행 때 덮어쓴다.

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is not None:
            self.abort()
