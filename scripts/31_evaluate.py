"""2-3: 학습된 모델 평가 — 클래스별 mAP + 조건별·크기별 재현율 + 신호등 색 혼동.

    python scripts/31_evaluate.py --run v2_base --dataset v2 --splits val test

결과: runs/detect/<run>/eval/<split>.json 과 콘솔 표
  - 재현율은 신뢰도 conf 이상 예측 기준 (실제 주행에서 쓸 임계값과 같은 조건)
  - 거리는 YOLO 라벨에 없으므로 박스 높이(px)로 대신 (멀수록 작다)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from driveloop.config import PROJECT_ROOT
from driveloop.eval.detection import match_image

TL_GROUP = {"tl_red": "traffic_light", "tl_yellow": "traffic_light", "tl_green": "traffic_light"}
SIZE_BINS = [0, 12, 16, 24, 40, 10_000]
SIZE_LABELS = ["<12", "12–16", "16–24", "24–40", "40+"]


def read_gt(txt: Path, names: dict, W: int, H: int) -> list[dict]:
    out = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append({"cls": names[int(c)], "box": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)})
    return out


def evaluate_split(model, ds: Path, manifest: pd.DataFrame, names: dict, split: str, conf: float,
                   imgsz: int, batch: int) -> dict:
    rows, fps = [], Counter()
    m = manifest[manifest.split == split].reset_index(drop=True)
    paths = [str(ds / p) for p in m.image]
    for start in range(0, len(paths), batch):
        results = model.predict(paths[start:start + batch], conf=conf, imgsz=imgsz, verbose=False)
        for r, meta in zip(results, m.iloc[start:start + batch].itertuples()):
            H, W = r.orig_shape
            gt = read_gt(ds / meta.image.replace("images/", "labels/", 1).replace(".jpg", ".txt"), names, W, H)
            pred = [{"cls": names[int(c)], "box": tuple(b), "conf": float(s)}
                    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            res = match_image(gt, pred, class_agnostic_groups=TL_GROUP)
            for g in res["gt"]:
                rows.append({"cls": g["cls"], "matched_cls": g["matched_cls"], "box_h": g["box"][3] - g["box"][1],
                             "weather": meta.weather, "time": meta.time, "map": meta.map})
            fps.update(p["cls"] for p in res["fp"])
    d = pd.DataFrame(rows)
    d["hit"] = d.matched_cls == d.cls                       # 위치 + 클래스(색)까지 맞음
    d["found"] = d.matched_cls.notna()                      # 위치는 찾음 (색은 틀릴 수 있음)
    d["size"] = pd.cut(d.box_h, SIZE_BINS, labels=SIZE_LABELS)
    d["cond"] = d.weather + "·" + d.time

    def recall_table(by):
        t = d.groupby(["cls", by], observed=True).hit.agg(recall="mean", n="size").reset_index()
        return {f"{r.cls}|{r[by]}": {"recall": round(r.recall, 3), "n": int(r.n)} for _, r in t.iterrows()}

    tl = d[d.cls.str.startswith("tl_")]
    confusion = pd.crosstab(tl.cls, tl.matched_cls.fillna("missed")).to_dict(orient="index")
    return {
        "split": split, "conf": conf, "images": len(m), "objects": len(d),
        "recall_by_class": {c: {"recall": round(g.hit.mean(), 3), "n": len(g)} for c, g in d.groupby("cls")},
        "false_positives": dict(fps),
        "recall_by_condition": recall_table("cond"),
        "recall_by_size": recall_table("size"),
        "tl_color_confusion": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in confusion.items()},
    }


def print_report(rep: dict, mapv: dict) -> None:
    print(f"\n===== {rep['split']} ({rep['images']}장, 정답 {rep['objects']}개, conf ≥ {rep['conf']}) =====")
    print(f"{'클래스':10} {'mAP50':>6} {'mAP50-95':>8} {'재현율':>6} {'정답수':>6} {'오검출':>6}")
    for c, v in rep["recall_by_class"].items():
        mp = mapv.get(c, {})
        print(f"{c:10} {mp.get('map50', 0):6.3f} {mp.get('map', 0):8.3f} {v['recall']:6.3f} {v['n']:6d} "
              f"{rep['false_positives'].get(c, 0):6d}")
    for title, key in (("조건별 재현율", "recall_by_condition"), ("크기(px)별 재현율", "recall_by_size")):
        t = pd.DataFrame([{"cls": k.split("|")[0], "g": k.split("|")[1], "v": f"{v['recall']:.2f} ({v['n']})"}
                          for k, v in rep[key].items()]).pivot(index="cls", columns="g", values="v")
        print(f"\n-- {title} [재현율 (개수)]\n{t.to_string()}")
    print("\n-- 신호등 색 혼동 (행=정답, 열=예측)")
    print(pd.DataFrame(rep["tl_color_confusion"]).T.fillna(0).astype(int).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v2_base")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--dataset", default="v2")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    run_dir = PROJECT_ROOT / "runs" / "detect" / args.run
    ds = PROJECT_ROOT / "data" / "datasets" / args.dataset
    data_yaml = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in data_yaml["names"].items()}
    manifest = pd.read_parquet(ds / "manifest.parquet")
    model = YOLO(str(run_dir / "weights" / args.weights))
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)

    for split in args.splits:
        v = model.val(data=str(ds / "data.yaml"), split=split, imgsz=args.imgsz, batch=args.batch,
                      plots=False, verbose=False, project=str(out), name=f"val_{split}", exist_ok=True)
        mapv = {names[i]: {"map50": round(float(v.box.ap50[j]), 3), "map": round(float(v.box.ap[j]), 3)}
                for j, i in enumerate(v.box.ap_class_index)}
        rep = evaluate_split(model, ds, manifest, names, split, args.conf, args.imgsz, args.batch)
        rep["map"] = mapv
        rep["weights"] = str(run_dir / "weights" / args.weights)
        (out / f"{split}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print_report(rep, mapv)


if __name__ == "__main__":
    main()
