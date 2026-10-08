"""'직접 해 보기' 페이지(docs/try.html) 에셋 — 내 모델을 방문자 브라우저에서 돌린다 (CPU만, CARLA 불필요).

    python scripts/42_build_try.py                     # 기본: v4_base, 예시 9장
    python scripts/42_build_try.py --run v4_base --force

출력 : docs/assets/try/model.onnx        학습 평가와 같은 1280×736 입력 (720p 사진을 패딩 16px만 붙여 그대로)
       docs/assets/try/samples/<id>.jpg   테스트셋(Town05, 학습에 안 쓴 도시) 예시 + 썸네일 <id>_t.jpg
       docs/assets/try/try.json           클래스 이름, 입력 크기, 예시별 조건·정답 박스
브라우저 쪽 전처리·후처리(레터박스 → 클래스별 NMS)는 이 스크립트의 check()와 같은 식이고,
check()가 ONNX 결과를 원본 .pt 결과와 비교해 변환을 검증한다.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from driveloop.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "assets" / "try"
W, H = 1280, 736
# (id, 테스트 이미지, 무엇을 보여 주나) — 날씨·시간 4조건을 고루, 마지막은 놓친 예
SAMPLES = [
    ("ped_cross", "Town05_clear_noon_high_r2_000068", "crossing"),
    ("red_truck", "Town05_clear_noon_high_r2_000044", "red"),
    ("rain_ped", "Town05_rain_noon_low_r2_000131", "crossing"),
    ("rain_rider", "Town05_rain_noon_low_r2_000002", "rider"),
    ("night_green", "Town05_clear_night_high_r0_000131", "green"),
    ("night_busy", "Town05_clear_night_high_r0_000013", "busy"),
    ("rainnight_ped", "Town05_rain_night_low_r2_000160", "crossing"),
    ("rainnight_red", "Town05_rain_night_low_r0_000055", "red"),
    ("rainnight_miss", "Town05_rain_night_low_r2_000052", "miss"),
]


def letterbox(img: Image.Image):
    w, h = img.size
    r = min(W / w, H / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (W - nw) // 2, (H - nh) // 2
    canvas = Image.new("RGB", (W, H), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0, r, px, py


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def decode(out, r, px, py, conf=0.25, iou_t=0.7):
    """(1, 4+C, N) → [(cls, score, xyxy)] — ultralytics와 같은 클래스별 NMS (try.js도 같은 식)."""
    p = out[0].T
    cls, sc = p[:, 4:].argmax(1), p[:, 4:].max(1)
    k = sc >= conf
    p, cls, sc = p[k], cls[k], sc[k]
    b = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    b = (b - [px, py, px, py]) / r
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-sc[idx])]
        while len(idx):
            keep.append(idx[0])
            idx = idx[1:][iou(b[idx[0]], b[idx[1:]]) < iou_t] if len(idx) > 1 else idx[1:]
    return [(int(cls[i]), float(sc[i]), b[i]) for i in keep]


def check(onnx_path: Path, pt_path: Path, images: list[Path]) -> None:
    """ONNX + 위 후처리 결과가 원본 .pt 예측과 같은지 (박스 IoU > 0.9, 같은 클래스, 신뢰도 차이)."""
    import onnxruntime as ort
    from ultralytics import YOLO

    sess, pt = ort.InferenceSession(str(onnx_path)), YOLO(str(pt_path))
    worst, miss = 0.0, 0
    for p in images:
        img = Image.open(p).convert("RGB")
        x, r, px, py = letterbox(img)
        mine = decode(sess.run(None, {"images": x})[0], r, px, py)
        ref = pt.predict(img, imgsz=W, conf=0.25, device="cpu", verbose=False)[0].boxes
        rb, rc, rs = ref.xyxy.numpy(), ref.cls.numpy().astype(int), ref.conf.numpy()
        miss += abs(len(mine) - len(rb))
        for c, s, b in mine:
            j = iou(b, rb).argmax() if len(rb) else None
            if j is None or iou(b, rb)[j] < 0.9 or rc[j] != c:
                miss += 1
            else:
                worst = max(worst, abs(s - rs[j]))
    print(f"[try] check: {len(images)}장, 불일치 {miss}, 최대 신뢰도 차이 {worst:.4f}")
    if miss or worst > 0.01:
        raise SystemExit("ONNX 결과가 .pt와 다릅니다")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="v4_base", help="runs/detect/<run>/weights/best.pt")
    ap.add_argument("--dataset", default="v4", help="예시 사진·정답 라벨을 가져올 데이터셋 (test split)")
    ap.add_argument("--force", action="store_true", help="모델을 다시 변환")
    args = ap.parse_args()

    weights = PROJECT_ROOT / "runs" / "detect" / args.run / "weights" / "best.pt"
    ds = PROJECT_ROOT / "data" / "datasets" / args.dataset
    (OUT / "samples").mkdir(parents=True, exist_ok=True)

    model = OUT / "model.onnx"
    if args.force or not model.exists():
        from ultralytics import YOLO

        # export는 가중치 옆에 파일을 만들므로 임시 복사본에서 변환
        tmp = OUT / f"_{args.run}.pt"
        shutil.copy(weights, tmp)
        try:
            out = YOLO(str(tmp)).export(format="onnx", imgsz=(H, W), device="cpu", simplify=True, opset=17)
            shutil.move(out, model)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"[try] model ← {args.run} ({model.stat().st_size / 1e6:.1f} MB)")

    import yaml

    names = list(yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"].values())

    samples = []
    for sid, stem, kind in SAMPLES:
        src = ds / "images" / "test" / f"{stem}.jpg"
        img = Image.open(src).convert("RGB")
        img.save(OUT / "samples" / f"{sid}.jpg", quality=85, optimize=True)
        img.resize((320, 180), Image.LANCZOS).save(OUT / "samples" / f"{sid}_t.jpg", quality=80, optimize=True)
        w, h = img.size
        gt = []
        for line in (ds / "labels" / "test" / f"{stem}.txt").read_text().splitlines():
            if line.strip():
                k, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                gt.append([int(k), round(cx - bw / 2, 1), round(cy - bh / 2, 1), round(cx + bw / 2, 1), round(cy + bh / 2, 1)])
        cond = "_".join(stem.split("_")[1:3])
        samples.append({"id": sid, "src": f"assets/try/samples/{sid}.jpg", "thumb": f"assets/try/samples/{sid}_t.jpg",
                        "cond": cond, "kind": kind, "frame": stem, "gt": gt})

    meta = {"run": args.run, "dataset": args.dataset, "names": names, "input": [W, H],
            "model": "assets/try/model.onnx", "model_mb": round(model.stat().st_size / 1e6, 1),
            "weights_mb": round(weights.stat().st_size / 1e6, 1), "samples": samples}
    (OUT / "try.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    check(model, weights, [ds / "images" / "test" / f"{s['frame']}.jpg" for s in samples])
    print(f"[try] {len(samples)} samples → {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
