"""2-2b: 원본 + 라벨 → frames.parquet / objects.parquet + QC 요약.

    python scripts/22_build_metadata.py --dataset v1

출력 (data/processed/<dataset>/):
  frames.parquet   프레임 1행 (조건 태그, 자차 상태, 객체 수, 밝기, dHash, QC 플래그)
  objects.parquet  라벨 1행 (박스, 거리, 각도, 불빛 색 검증, QC 플래그)
  qc_report.json   플래그별 개수, 사용한 규칙
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PIL import Image

from driveloop.config import CONFIG_DIR, PROJECT_ROOT
from driveloop.data.metadata import QCRules, frame_row, load_qc_rules, object_row


def process_episode(raw_ep: Path, label_path: Path, qc: QCRules) -> tuple[list[dict], list[dict]]:
    episode = json.loads((raw_ep / "episode.json").read_text(encoding="utf-8"))
    W, H = episode["camera"]["width"], episode["camera"]["height"]
    labels = {}
    for line in label_path.open(encoding="utf-8"):
        rec = json.loads(line)
        labels[rec["index"]] = rec
    frames, objects = [], []
    for line in (raw_ep / "frames.jsonl").open(encoding="utf-8"):
        frame = json.loads(line)
        lab = labels[frame["index"]]
        try:
            img = Image.open(raw_ep / frame["rgb"]).convert("RGB")
        except Exception as e:  # 손상 파일도 행은 남기고 플래그
            frames.append({"episode_id": episode["episode_id"], "index": frame["index"],
                           "qc_flags": f"unreadable_image:{type(e).__name__}", "qc_pass": False})
            continue
        objs = [object_row(o, img, W, H, qc) for o in lab["objects"]]
        frames.append(frame_row(frame, lab, episode, img, objs, qc))
        for o in objs:
            objects.append({"episode_id": episode["episode_id"], "index": frame["index"],
                            "map": episode["tags"]["map"], "weather": episode["tags"]["weather"],
                            "time": episode["tags"]["time"], **o})
    return frames, objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--qc", default=str(CONFIG_DIR / "qc.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    raw = PROJECT_ROOT / "data" / "raw" / args.dataset
    out = PROJECT_ROOT / "data" / "processed" / args.dataset
    label_dir = out / "labels"
    qc = load_qc_rules(args.qc)
    episodes = sorted(p.stem for p in label_dir.glob("*.jsonl"))
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process_episode, [raw / e for e in episodes],
                                [label_dir / f"{e}.jsonl" for e in episodes], [qc] * len(episodes)))
    frames = pd.DataFrame([r for f, _ in results for r in f])
    objects = pd.DataFrame([r for _, o in results for r in o])
    frames.to_parquet(out / "frames.parquet", index=False)
    objects.to_parquet(out / "objects.parquet", index=False)

    def flag_counts(df):
        s = df["qc_flags"].fillna("").str.split(",").explode()
        return s[s != ""].value_counts().to_dict()

    label_summary = json.loads((label_dir / "_summary.json").read_text(encoding="utf-8"))
    report = {
        "dataset": args.dataset, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_rules_version": label_summary["rules"]["version"], "qc_rules": asdict(qc),
        "frames": len(frames), "frames_qc_pass": int(frames["qc_pass"].sum()),
        "frame_flags": flag_counts(frames), "objects": len(objects), "object_flags": flag_counts(objects),
        "lamp_color": objects["lamp_color"].value_counts().to_dict(),
    }
    (out / "qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[meta] {len(frames)} frames, {len(objects)} objects, {time.time() - start:.0f}s → {out}")
    print(f"       QC 통과 프레임 {report['frames_qc_pass']}/{len(frames)}")
    print(f"       프레임 플래그: {report['frame_flags']}")
    print(f"       객체 플래그  : {report['object_flags']}")
    print(f"       신호등 불빛 색: {report['lamp_color']}")


if __name__ == "__main__":
    main()
