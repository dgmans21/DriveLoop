"""2-3: YOLO 학습. 회차로 나눠 실행할 수 있다.

    python scripts/30_train.py --config configs/train/v2_base.yaml --stop-after 10   # 1회차: 10 epoch 후 멈춤
    python scripts/30_train.py --config configs/train/v2_base.yaml --resume --stop-after 10  # 이어서 10 epoch
    python scripts/30_train.py --config configs/train/v2_base.yaml --resume          # 끝까지

결과: runs/detect/<name>/ (ultralytics 기본 산출물)
     + driveloop_meta.json — 데이터셋 해시(dvc.lock)·git 커밋·설정·회차 기록 (계보 추적)

--stop-after: ultralytics는 epoch 끝에 last.pt를 저장한 '뒤' on_fit_epoch_end 콜백을 부른다.
  그 콜백에서 프로세스를 바로 종료해야 last.pt가 이어서 학습 가능한 상태로 남는다
  (trainer.stop으로 멈추면 final_eval이 last.pt의 optimizer를 지워 resume이 안 된다).
"""
import argparse
import json
import os
import subprocess
import sys
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


def write_meta(run_dir: Path, cfg: dict, config_path: str, session: dict) -> None:
    path = run_dir / "driveloop_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "config": cfg, "config_path": config_path, "dataset_md5": dataset_hash(cfg["dataset"]), "sessions": []}
    meta["sessions"].append(session)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="runs/detect/<name>/weights/last.pt 에서 이어서")
    parser.add_argument("--stop-after", type=int, default=None, help="이번 회차에 N epoch 돈 뒤 멈춤 (resume 가능)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from ultralytics import YOLO  # 무거운 import는 설정 확인 후

    run_dir = PROJECT_ROOT / "runs" / "detect" / cfg["name"]
    session = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(),
               "resume": args.resume, "stop_after": args.stop_after}
    start = time.time()
    done_this_session = {"n": 0, "first": None}

    def on_epoch_end(trainer) -> None:
        done_this_session["n"] += 1
        if done_this_session["first"] is None:
            done_this_session["first"] = trainer.epoch + 1
        if args.stop_after and done_this_session["n"] >= args.stop_after and trainer.epoch + 1 < trainer.epochs:
            session.update(epochs=[done_this_session["first"], trainer.epoch + 1],
                           seconds=round(time.time() - start, 1), ended="stopped (resumable)")
            write_meta(Path(trainer.save_dir), cfg, args.config, session)
            print(f"\n[train] {done_this_session['n']} epoch 완료 (epoch {trainer.epoch + 1}/{trainer.epochs}) "
                  f"→ 멈춤. 이어서: --resume", flush=True)
            sys.stdout.flush()
            os._exit(0)  # last.pt는 이미 저장됨. 정상 종료 경로를 타면 resume 불가 상태가 된다

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"이어서 학습할 체크포인트가 없음: {last}")
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        print(f"[train] resume {cfg['name']} from {last}")
        model.train(resume=True)
    else:
        if run_dir.exists():
            raise SystemExit(f"이미 있는 실행: {run_dir} (이어서 하려면 --resume)")
        model = YOLO(cfg["model"])
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        print(f"[train] {cfg['name']}: dataset={cfg['dataset']} ({dataset_hash(cfg['dataset'])}) git={session['git_commit']}")
        model.train(
            data=str(PROJECT_ROOT / "data" / "datasets" / cfg["dataset"] / "data.yaml"),
            epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"], workers=cfg["workers"],
            seed=cfg["seed"], device=cfg["device"], project=str(run_dir.parent), name=cfg["name"],
            exist_ok=False, **cfg.get("extra", {}),
        )
    session.update(seconds=round(time.time() - start, 1), ended="finished")
    write_meta(Path(model.trainer.save_dir), cfg, args.config, session)
    print(f"[train] 학습 종료 ({session['seconds']:.0f}s) → {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
