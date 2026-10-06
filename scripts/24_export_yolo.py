"""2-2d: data/processed/<source> → data/datasets/<name> (YOLO 형식).

    python scripts/24_export_yolo.py --config configs/export/v1.yaml

출력:
  images/{train,val}/<episode>_<index>.jpg   무시 영역을 회색으로 가린 이미지
  labels/{train,val}/<episode>_<index>.txt   'cls cx cy w h'
  data.yaml                                  ultralytics 학습 설정
  manifest.parquet                           이미지별 출처·조건·객체 수 (조건별 평가에 사용)
  export_report.json                         split/클래스/무시 사유별 개수, 사용한 설정과 규칙 버전
"""
import argparse
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from driveloop.config import CONFIG_DIR, PROJECT_ROOT
from driveloop.data.export import apply_ignore_mask, load_export_config, yolo_line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "export" / "v1.yaml"))
    args = parser.parse_args()
    cfg = load_export_config(args.config)

    raw = PROJECT_ROOT / "data" / "raw" / cfg.source
    proc = PROJECT_ROOT / "data" / "processed" / cfg.source
    out = PROJECT_ROOT / "data" / "datasets" / cfg.name
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    frames = pd.read_parquet(proc / "frames.parquet")
    objects = pd.read_parquet(proc / "objects.parquet")
    if cfg.use_dedup:
        dedup = pd.read_parquet(proc / "dedup.parquet")[["episode_id", "index", "keep"]]
        frames = frames.merge(dedup, on=["episode_id", "index"])
        frames = frames[frames.keep]
    flags = frames.qc_flags.fillna("").str.split(",")
    frames = frames[~flags.apply(lambda fs: any(f in cfg.drop_frame_flags for f in fs))]

    # 객체별 QC 결과 (라벨 jsonl 의 객체와 episode/index/actor/cls/좌표로 매칭)
    obj_key = ["episode_id", "index", "actor_id", "cls", "x1", "y1"]
    obj_qc = objects.set_index(obj_key)[["qc_flags", "lamp_color"]].to_dict("index")
    cls_id = {c: i for i, c in enumerate(cfg.classes)}
    val = set(cfg.val_episodes)
    missing_val = val - set(frames.episode_id)
    if missing_val:
        raise SystemExit(f"val_episodes 에 없는 에피소드: {sorted(missing_val)}")

    labels_by_ep = {}
    counts = {"train": Counter(), "val": Counter()}
    ignored = Counter()
    rows = []
    for ep, g in frames.groupby("episode_id"):
        if ep not in labels_by_ep:
            labels_by_ep[ep] = {json.loads(l)["index"]: json.loads(l)
                                for l in (proc / "labels" / f"{ep}.jsonl").open(encoding="utf-8")}
        split = "val" if ep in val else "train"
        for fr in g.itertuples():
            lab = labels_by_ep[ep][fr.index]
            img = np.asarray(Image.open(raw / ep / fr.rgb).convert("RGB"))
            H, W = img.shape[:2]
            keep, ignore = [], [i["bbox"] for i in lab["ignores"]]
            ignored.update(f"{i['cls']}:{i['reason']}" for i in lab["ignores"])
            for o in lab["objects"]:
                q = obj_qc.get((ep, fr.index, o["actor_id"], o["cls"], o["bbox"][0], o["bbox"][1]), {})
                obj_flags = (q.get("qc_flags") or "").split(",")
                if any(f in cfg.ignore_object_flags for f in obj_flags):
                    ignore.append(o["bbox"]); ignored[f"{o['cls']}:qc_flag"] += 1
                elif q.get("lamp_color") in cfg.ignore_lamp_color:
                    ignore.append(o["bbox"]); ignored[f"{o['cls']}:lamp_{q['lamp_color']}"] += 1
                else:
                    keep.append(o)
            img = apply_ignore_mask(img, ignore, [o["bbox"] for o in keep], cfg.ignore_fill, cfg.ignore_pad_px)
            stem = f"{ep}_{fr.index:06d}"
            Image.fromarray(img).save(out / "images" / split / f"{stem}.jpg", quality=cfg.jpeg_quality)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(yolo_line(cls_id[o["cls"]], o["bbox"], W, H) for o in keep), encoding="utf-8")
            c = Counter(o["cls"] for o in keep)
            counts[split].update(c)
            rows.append({"split": split, "image": f"images/{split}/{stem}.jpg", "episode_id": ep,
                         "index": fr.index, "map": fr.map, "weather": fr.weather, "time": fr.time,
                         "traffic": fr.traffic, "ego_stopped": fr.ego_stopped, "n_ignore": len(ignore),
                         **{f"n_{k}": c.get(k, 0) for k in cfg.classes}})

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(out / "manifest.parquet", index=False)
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "names": {i: c for i, c in enumerate(cfg.classes)},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    label_rules = json.loads((proc / "labels" / "_summary.json").read_text(encoding="utf-8"))["rules"]
    report = {
        "dataset": cfg.name, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": asdict(cfg),
        "label_rules_version": label_rules["version"],
        "images": manifest.split.value_counts().to_dict(),
        "objects": {s: {k: counts[s].get(k, 0) for k in cfg.classes} for s in counts},
        "ignored": dict(ignored.most_common()),
    }
    (out / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[export] {len(manifest)} images → {out}")
    for s in ("train", "val"):
        print(f"  {s:5} images={report['images'].get(s, 0):5}  " +
              "  ".join(f"{k}={counts[s].get(k, 0)}" for k in cfg.classes))
    print(f"  무시 영역: {dict(ignored.most_common())}")


if __name__ == "__main__":
    main()
