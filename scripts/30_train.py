"""2-3: YOLO 학습.

    python scripts/30_train.py --config configs/train/v2_trial.yaml

결과: runs/detect/<name>/ (ultralytics 기본 산출물)
     + runs/detect/<name>/driveloop_meta.json — 학습에 쓴 데이터셋 해시(dvc.lock)·git 커밋·설정 (계보 추적)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

from driveloop.config import PROJECT_ROOT


def dataset_hash(dataset: str) -> str | None:
    """dvc.lock에서 data/datasets/<dataset> 출력의 md5."""
    lock = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8"))
    for stage in lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == f"data/datasets/{dataset}":
                return out.get("md5")
    return None


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from ultralytics import YOLO  # 무거운 import는 설정 확인 후

    data_yaml = PROJECT_ROOT / "data" / "datasets" / cfg["dataset"] / "data.yaml"
    project = PROJECT_ROOT / "runs" / "detect"
    meta = {
        "config": cfg, "config_path": str(Path(args.config)),
        "dataset_md5": dataset_hash(cfg["dataset"]), "git_commit": git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"[train] {cfg['name']}: dataset={cfg['dataset']} ({meta['dataset_md5']}) git={meta['git_commit']}")

    model = YOLO(cfg["model"])
    start = time.time()
    model.train(
        data=str(data_yaml), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        workers=cfg["workers"], seed=cfg["seed"], device=cfg["device"],
        project=str(project), name=cfg["name"], exist_ok=False, **cfg.get("extra", {}),
    )
    meta["train_seconds"] = round(time.time() - start, 1)
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "driveloop_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] done in {meta['train_seconds']:.0f}s → {save_dir}")


if __name__ == "__main__":
    main()
