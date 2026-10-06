"""2-2c: frames.parquet → dedup.parquet (프레임별 keep / dup_of / 판단 근거).

    python scripts/23_dedup.py --dataset v1
    python scripts/23_dedup.py --dataset v1 --samples 6   # 지운 프레임 vs 대표 프레임 비교 이미지

원본·라벨은 지우지 않는다. 내보내기 단계(2-2d)가 keep=True 프레임만 쓴다.
"""
import argparse
import json
import random
import time
from dataclasses import asdict

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from driveloop.config import CONFIG_DIR, PROJECT_ROOT
from driveloop.data.dedup import LABEL_KEYS, dedup_episode, load_dedup_rules


def sample_sheet(frames: pd.DataFrame, dedup: pd.DataFrame, raw, n: int, path) -> None:
    """지운 프레임(오른쪽)과 그 대표 프레임(왼쪽)을 나란히."""
    removed = dedup[~dedup.keep]
    rng = random.Random(0)
    picks = removed.iloc[rng.sample(range(len(removed)), min(n, len(removed)))]
    font = ImageFont.truetype("arial.ttf", 16)
    W, H = 480, 270
    out = Image.new("RGB", (2 * W + 10, len(picks) * (H + 28)), (30, 30, 30))
    d = ImageDraw.Draw(out)
    for row, (_, r) in enumerate(picks.iterrows()):
        y = row * (H + 28)
        for col, idx in enumerate((int(r.dup_of), int(r["index"]))):
            img = Image.open(raw / r.episode_id / "rgb" / f"{idx:06d}.jpg").resize((W, H))
            out.paste(img, (col * (W + 10), y + 24))
        d.text((4, y + 4), f"{r.episode_id}  keep #{int(r.dup_of)}  |  removed #{int(r['index'])} "
                           f"(hamming {int(r.hamming)}, move {r.move_m:.2f}m)", fill=(255, 255, 0), font=font)
    out.save(path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--rules", default=str(CONFIG_DIR / "dedup.yaml"))
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
    raw = PROJECT_ROOT / "data" / "raw" / args.dataset
    rules = load_dedup_rules(args.rules)
    frames = pd.read_parquet(out_dir / "frames.parquet").sort_values(["episode_id", "index"])

    rows = []
    for ep, g in frames.groupby("episode_id"):
        for r in dedup_episode(g.to_dict("records"), rules):
            rows.append({"episode_id": ep, **r})
    dedup = pd.DataFrame(rows)
    dedup.to_parquet(out_dir / "dedup.parquet", index=False)

    m = frames.merge(dedup[["episode_id", "index", "keep"]], on=["episode_id", "index"])
    kept = m[m.keep]
    by_cond = m.groupby(["map", "weather", "time", "traffic"]).agg(before=("keep", "size"), after=("keep", "sum"))
    cls_before = {k: int(m[k].sum()) for k in LABEL_KEYS}
    cls_after = {k: int(kept[k].sum()) for k in LABEL_KEYS}
    report = {
        "dataset": args.dataset, "rules": asdict(rules), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames_before": len(m), "frames_after": int(m.keep.sum()),
        "reasons": dedup.reason.value_counts().to_dict(),
        "stopped_frames_before": int(m.ego_stopped.sum()), "stopped_frames_after": int(kept.ego_stopped.sum()),
        "objects_before": cls_before, "objects_after": cls_after,
    }
    (out_dir / "dedup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[dedup] 프레임 {len(m)} → {int(m.keep.sum())} ({m.keep.mean() * 100:.0f}% 유지)")
    print(f"        정지 프레임 {report['stopped_frames_before']} → {report['stopped_frames_after']}")
    print(f"        판단 근거: {report['reasons']}")
    print("        객체: " + "  ".join(f"{k[2:]} {cls_before[k]}→{cls_after[k]}" for k in LABEL_KEYS))
    print(by_cond.assign(kept_pct=(by_cond.after / by_cond.before * 100).round(0)).to_string())
    if args.samples:
        path = PROJECT_ROOT / "outputs" / "dedup_samples.jpg"
        sample_sheet(frames, dedup.merge(frames[["episode_id", "index"]]), raw, args.samples, path)
        print(f"        비교 이미지 → {path}")


if __name__ == "__main__":
    main()
