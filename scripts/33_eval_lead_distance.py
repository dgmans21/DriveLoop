"""B-4: 카메라 거리 추정 검증 — 자동 라벨의 실제 거리(depth)와 비교 (CARLA·GPU 불필요).

    python scripts/33_eval_lead_distance.py                  # 원본 v1~v3 차량 라벨
    python scripts/33_eval_lead_distance.py --sources v3

대상: '앞차 후보'처럼 보이는 차량 라벨만
  - 화면 가운데 근처 (박스 중심 x가 화면 폭의 30~70%), 박스 아래 변이 화면 안 (잘리지 않음)
  - 많이 가려지지 않음 (visible_ratio ≥ 0.5) — 바닥선이 다른 차에 가려지면 바닥선 방법이 틀어진다
라벨 depth = 카메라 좌표에서 그 차 3D 박스의 가장 가까운 꼭짓점까지 앞쪽 거리 (차 뒷면).
카메라 높이 H는 라벨로 보정(중앙값)하고, 설정값(1.7m, 차량 원점 기준)과 비교해 본다.
출력: outputs/lead_distance/report.json + 콘솔 표 (거리 구간별 오차)
"""
import argparse
import json

import numpy as np
import pandas as pd

from driveloop.config import PROJECT_ROOT
from driveloop.data.collection_config import CaptureCamera
from driveloop.data.snapshot import camera_intrinsics
from driveloop.perception.mono_distance import ground_distance, width_distance

BINS = [0, 10, 20, 30, 40, 60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["v1", "v2", "v3"])
    args = ap.parse_args()

    cam = CaptureCamera()
    K = np.array(camera_intrinsics(cam.width, cam.height, cam.fov))
    fx, fy, cy = K[0, 0], K[1, 1], K[1, 2]
    frames = []
    for s in args.sources:
        o = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / s / "objects.parquet")
        o["source"] = s
        frames.append(o)
    o = pd.concat(frames, ignore_index=True)
    v = o[(o.cls == "vehicle") & o.depth.notna()].copy()
    cx_rel = ((v.x1 + v.x2) / 2) / cam.width
    v = v[(cx_rel.between(0.3, 0.7)) & (v.y2 < cam.height - 2) & (v.visible_ratio.fillna(0) >= 0.5)
          & (v.depth > 2.0)].copy()

    # 카메라 높이 보정: H = depth · (y2 − cy) / fy 의 중앙값 (가까운 차일수록 바닥선이 정확)
    near = v[v.depth < 25]
    h_fit = float(np.median(near.depth * (near.y2 - cy) / fy))
    # 차 폭 보정: W = depth · box_w / fx 의 중앙값 (정면에 가까운 차만: 화면 정중앙 ±10%)
    center = v[(((v.x1 + v.x2) / 2) / cam.width).between(0.4, 0.6)]
    w_fit = float(np.median(center.depth * center.box_w / fx))

    v["z_ground_cfg"] = [ground_distance(y, fy, cy, cam.z) for y in v.y2]
    v["z_ground"] = [ground_distance(y, fy, cy, h_fit) for y in v.y2]
    v["z_width"] = [width_distance(w, fx, w_fit) for w in v.box_w]
    v["bin"] = pd.cut(v.depth, BINS)

    rows = []
    for name in ("z_ground_cfg", "z_ground", "z_width"):
        err = (v[name] - v.depth)
        rel = err.abs() / v.depth
        g = pd.DataFrame({"bin": v["bin"], "abs": err.abs(), "rel": rel, "bias": err})
        t = g.groupby("bin", observed=True).agg(n=("abs", "size"), mae=("abs", "median"),
                                               p90=("abs", lambda x: x.quantile(0.9)),
                                               rel_med=("rel", "median"), bias=("bias", "median")).round(2)
        t.insert(0, "method", name)
        rows.append(t.reset_index())
    table = pd.concat(rows)
    table["bin"] = table["bin"].astype(str)

    out = PROJECT_ROOT / "outputs" / "lead_distance"
    out.mkdir(parents=True, exist_ok=True)
    report = {"n": int(len(v)), "cam_height_cfg": cam.z, "cam_height_fit": round(h_fit, 3),
              "car_width_fit": round(w_fit, 3), "fx": fx, "fy": fy, "cy": cy,
              "table": table.to_dict(orient="records")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 160)
    print(f"[lead-distance] 앞차 후보 라벨 {len(v)}개 ({', '.join(args.sources)})")
    print(f"  카메라 높이: 설정 {cam.z} m → 라벨로 보정 {h_fit:.3f} m / 차 폭 보정 {w_fit:.2f} m")
    print("  (mae = 절대 오차 중앙값 m, p90 = 90% 오차 m, rel_med = 상대 오차 중앙값, bias = 부호 있는 오차 중앙값)\n")
    print(table.to_string(index=False))
    print(f"\n[lead-distance] → {out / 'report.json'}")


if __name__ == "__main__":
    main()
