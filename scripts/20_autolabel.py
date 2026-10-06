"""2-2a: 원본(data/raw/<dataset>) → 2D 박스 라벨(data/processed/<dataset>/labels).

    python scripts/20_autolabel.py --dataset v1
    python scripts/20_autolabel.py --dataset v1 --rules configs/labeling.yaml --workers 4

에피소드별로 labels/<episode_id>.jsonl (한 줄 = 한 프레임) 를 쓰고,
사용한 규칙과 클래스별 개수를 labels/_summary.json 에 남긴다.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from driveloop.config import CONFIG_DIR, PROJECT_ROOT
from driveloop.data.autolabel import LabelRules, find_ego_instance, label_frame_full, load_rules
from driveloop.data.writer import is_complete


def label_episode(ep_dir: Path, out_path: Path, rules: LabelRules) -> dict:
    episode = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    counts = Counter()
    frames = 0
    with open(ep_dir / "frames.jsonl", encoding="utf-8") as src:
        recs = [json.loads(line) for line in src]
    ego_id = recs[0]["ego"].get("id")
    if ego_id is None:  # v1: 자차 ID 미기록 → 보닛 위치로 판정
        ego_id = find_ego_instance([np.asarray(Image.open(ep_dir / r["seg"])) for r in recs[::15]])
    tmp = out_path.with_suffix(".jsonl.partial")
    with open(ep_dir / "frames.jsonl", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            frame = json.loads(line)
            seg = np.asarray(Image.open(ep_dir / frame["seg"]))
            objects, ignores = label_frame_full(frame, episode, seg, rules, ego_id)
            counts.update(o["cls"] for o in objects)
            counts.update(f"ignore_{i['cls']}" for i in ignores)
            dst.write(json.dumps({"index": frame["index"], "rgb": frame["rgb"], "objects": objects,
                                  "ignores": ignores}) + "\n")
            frames += 1
    tmp.replace(out_path)
    return {"episode_id": ep_dir.name, "frames": frames, "counts": dict(counts), "ego_id": ego_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--rules", default=str(CONFIG_DIR / "labeling.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    raw = PROJECT_ROOT / "data" / "raw" / args.dataset
    out = PROJECT_ROOT / "data" / "processed" / args.dataset / "labels"
    out.mkdir(parents=True, exist_ok=True)
    rules = load_rules(args.rules)
    episodes = sorted(p for p in raw.iterdir() if p.is_dir() and is_complete(p))
    print(f"[label] {len(episodes)} episodes, rules v{rules.version}, workers={args.workers}")

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(label_episode, ep, out / f"{ep.name}.jsonl", rules) for ep in episodes]
        results = [f.result() for f in futures]

    total = Counter()
    for r in results:
        total.update(r["counts"])
        print(f"  {r['episode_id']:32} " + "  ".join(f"{c}={r['counts'].get(c, 0):4}" for c in rules.classes))
    frames = sum(r["frames"] for r in results)
    summary = {"dataset": args.dataset, "rules": asdict(rules), "frames": frames,
               "counts": {c: total.get(c, 0) for c in rules.classes}, "episodes": results,
               "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {frames} frames, {time.time() - start:.0f}s → {out}")
    print("       " + "  ".join(f"{c}={total.get(c, 0)}" for c in rules.classes))


if __name__ == "__main__":
    main()
