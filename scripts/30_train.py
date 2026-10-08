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
    parser.add_argument("--workers", type=int, default=None,
                        help="데이터 로더 작업자 수 덮어쓰기 (resume에서도 가능, 결과 아닌 속도·메모리만 바뀜). "
                             "v4_base: 4개로 4 epoch 중 RAM 부족(MemoryError) → 3")
    parser.add_argument("--val-workers", type=int, default=None,
                        help="검증 로더를 '검증할 때만 뜨는' 작업자 N개로 교체 (기본: ultralytics가 학습 작업자×2를 "
                             "학습 내내 상주시킴 — Windows에서 작업자당 커밋 ~2GB, v4_base에서 4개 상주 = 8GB)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from ultralytics import YOLO  # 무거운 import는 설정 확인 후

    run_dir = PROJECT_ROOT / "runs" / "detect" / cfg["name"]
    session = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(),
               "resume": args.resume, "stop_after": args.stop_after, "workers": args.workers,
               "val_workers": args.val_workers}
    start = time.time()
    done_this_session = {"n": 0, "first": None}

    def swap_val_loader(trainer) -> None:
        """검증 로더 → 비상주(persistent_workers=False) DataLoader. 검증 결과는 같고, 작업자는 검증 중에만 존재."""
        import torch

        old = trainer.test_loader
        new = torch.utils.data.DataLoader(old.dataset, batch_size=old.batch_size, shuffle=False,
                                          num_workers=args.val_workers, persistent_workers=False,
                                          pin_memory=True, collate_fn=old.collate_fn)
        if hasattr(old, "close"):
            old.close()                       # 상주 작업자 종료
        trainer.test_loader = new
        trainer.validator.dataloader = new
        print(f"[train] 검증 로더 교체: 작업자 {args.val_workers}개, 검증 중에만", flush=True)

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
        if args.val_workers is not None:
            model.add_callback("on_pretrain_routine_end", swap_val_loader)
        print(f"[train] resume {cfg['name']} from {last}")
        model.train(resume=True, **({"workers": args.workers} if args.workers is not None else {}))
    else:
        if run_dir.exists():
            raise SystemExit(f"이미 있는 실행: {run_dir} (이어서 하려면 --resume)")
        model = YOLO(cfg["model"])
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        if args.val_workers is not None:
            model.add_callback("on_pretrain_routine_end", swap_val_loader)
        print(f"[train] {cfg['name']}: dataset={cfg['dataset']} ({dataset_hash(cfg['dataset'])}) git={session['git_commit']}")
        model.train(
            data=str(PROJECT_ROOT / "data" / "datasets" / cfg["dataset"] / "data.yaml"),
            epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
            workers=cfg["workers"] if args.workers is None else args.workers,
            seed=cfg["seed"], device=cfg["device"], project=str(run_dir.parent), name=cfg["name"],
            exist_ok=False, **cfg.get("extra", {}),
        )
    session.update(seconds=round(time.time() - start, 1), ended="finished")
    write_meta(Path(model.trainer.save_dir), cfg, args.config, session)
    print(f"[train] 학습 종료 ({session['seconds']:.0f}s) → {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
