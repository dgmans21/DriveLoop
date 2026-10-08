"""2-2e: 데이터셋 분포 리포트 (HTML + JSON 지표) — 무엇이 부족한지 수치로 찾는다.

    python scripts/25_distribution_report.py --dataset v2

입력: data/datasets/<dataset>/ (manifest.parquet, labels/*.txt, export_report.json)
출력: data/reports/<dataset>/distribution.html, distribution_report.json (dvc metrics)
"""
import argparse
import html
import json
from pathlib import Path

import pandas as pd
import yaml

from driveloop.config import CONFIG_DIR, PROJECT_ROOT

# 색: 의미 대응(차량=파랑, 신호 색=같은 색, 보행자=보라). 빨강·노랑이 이웃하지 않게 이 순서 고정 (팔레트 검증 통과 순서)
# 데이터셋에 없는 클래스(v1~v3의 보행자)는 빼고 그린다 → 없는 클래스가 '부족'으로 잡히지 않게
ALL_CLASSES = ["vehicle", "tl_red", "tl_green", "tl_yellow", "pedestrian"]
CLASS_ORDER = list(ALL_CLASSES)
CLASS_LABEL = {"vehicle": "차량", "tl_red": "빨간불", "tl_green": "초록불", "tl_yellow": "노란불", "pedestrian": "보행자"}
SIZE_BINS = [0, 8, 12, 16, 24, 40, 10_000]
SIZE_LABELS = ["<8", "8–12", "12–16", "16–24", "24–40", "40+"]
SPLIT_ORDER = ["train", "val", "test"]
TOTAL_KEY = {"train": "train_per_class", "val": "val_per_class", "test": "test_per_class"}
CELL_KEY = {"train": "cell_train", "val": "cell_val", "test": "cell_test"}


def load_objects(ds: Path, manifest: pd.DataFrame, names: list[str], img_h: int) -> pd.DataFrame:
    rows = []
    for r in manifest.itertuples():
        txt = ds / r.image.replace("images/", "labels/", 1).replace(".jpg", ".txt")
        for line in txt.read_text(encoding="utf-8").splitlines():
            c, _, _, _, h = line.split()
            rows.append({"split": r.split, "cls": names[int(c)], "map": r.map, "weather": r.weather,
                         "time": r.time, "box_h": float(h) * img_h})
    return pd.DataFrame(rows)


def analyze(objs: pd.DataFrame, manifest: pd.DataFrame, export_report: dict, cfg: dict,
            all_maps: list[str]) -> dict:
    t = cfg["targets"]
    splits = [s for s in SPLIT_ORDER if s in set(manifest.split)]
    split_cls = objs.groupby(["cls", "split"]).size().unstack(fill_value=0).reindex(CLASS_ORDER, fill_value=0)
    for s in splits:
        if s not in split_cls:
            split_cls[s] = 0
    cond = (objs.assign(cond=objs.weather + "·" + objs.time)
            .groupby(["cond", "cls", "split"]).size().unstack(fill_value=0))
    tl = objs[objs.cls.str.startswith("tl_")]
    small = float((tl.box_h < cfg["small_tl_px"]).mean()) if len(tl) else 0.0

    gaps = []
    for c in CLASS_ORDER:
        for s in splits:
            key = TOTAL_KEY[s]
            n = int(split_cls.loc[c, s])
            if n < t[key]:
                gaps.append({"kind": "class_total", "cls": c, "split": s, "have": n, "target": t[key],
                             "text": f"{CLASS_LABEL[c]} {s} {n}개 < 목표 {t[key]}"})
    conds = sorted({f"{w}·{tm}" for w in manifest.weather.unique() for tm in manifest.time.unique()})
    for cd in conds:
        for c in CLASS_ORDER:
            for s in splits:
                key = CELL_KEY[s]
                n = int(cond.loc[(cd, c), s]) if (cd, c) in cond.index and s in cond else 0
                if n < t[key]:
                    gaps.append({"kind": "cell", "cond": cd, "cls": c, "split": s, "have": n, "target": t[key],
                                 "text": f"{cd} · {CLASS_LABEL[c]} {s} {n}개 < {t[key]}"})
    train_maps = set(manifest.loc[manifest.split == "train", "map"])
    heldout = [m for m in all_maps if m not in train_maps]
    if len(heldout) < t["heldout_maps"]:
        gaps.append({"kind": "heldout_map", "have": len(heldout), "target": t["heldout_maps"],
                     "text": f"평가 전용 맵 {len(heldout)}개 < {t['heldout_maps']} (val 맵이 모두 학습에도 쓰임)"})

    return {
        "dataset": export_report["dataset"],
        "splits": splits,
        "images": export_report["images"],
        "objects": {c: {s: int(split_cls.loc[c, s]) for s in splits} for c in CLASS_ORDER},
        "tl_small_ratio": round(small, 3),
        "tl_size_hist": {c: dict(zip(SIZE_LABELS, pd.cut(tl[tl.cls == c].box_h, SIZE_BINS, labels=SIZE_LABELS)
                                     .value_counts().reindex(SIZE_LABELS, fill_value=0).astype(int).tolist()))
                         for c in CLASS_ORDER if c.startswith("tl_")},
        "ignored": export_report["ignored"],
        "num_gaps": len(gaps),
        "gaps": gaps,
        "_cond": {f"{cd}|{c}|{s}": int(v) for (cd, c), row in cond.iterrows() for s, v in row.items()},
        "_conds": conds,
    }


# ---------------- HTML (인라인 SVG, 외부 의존 없음) ----------------

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#7a7974;--grid:#e4e3df;--gap-ring:#fcfcfb;
 --c-vehicle:#2a78d6;--c-tl_red:#e34948;--c-tl_green:#008300;--c-tl_yellow:#eda100;--c-pedestrian:#8b5cf6;
 --seq-0:#f3f2ef;--seq-1:#cde2fb;--seq-2:#9ec5f4;--seq-3:#6da7ec;--seq-4:#3987e5;--seq-5:#256abf;--seq-6:#184f95;
 --warn:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--surface-2:#242422;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#9a998f;
 --grid:#383835;--gap-ring:#1a1a19;--c-vehicle:#3987e5;--c-tl_red:#e66767;--c-tl_green:#008300;--c-tl_yellow:#c98500;--c-pedestrian:#9b72f2;
 --seq-0:#242422;--seq-1:#104281;--seq-2:#184f95;--seq-3:#1c5cab;--seq-4:#256abf;--seq-5:#2a78d6;--seq-6:#3987e5;--warn:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--text-primary:#fff;
 --text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#383835;--gap-ring:#1a1a19;--c-vehicle:#3987e5;--c-tl_red:#e66767;
 --c-tl_green:#008300;--c-tl_yellow:#c98500;--c-pedestrian:#9b72f2;--seq-0:#242422;--seq-1:#104281;--seq-2:#184f95;--seq-3:#1c5cab;
 --seq-4:#256abf;--seq-5:#2a78d6;--seq-6:#3987e5;--warn:#e66767}
body{margin:0;background:var(--surface-1)}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.5 system-ui,"Malgun Gothic",sans-serif;
 max-width:980px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 4px}.sub{color:var(--text-secondary);margin:0 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface-2);border-radius:8px;padding:12px}.tile .v{font-size:24px;font-weight:600}
.tile .k{color:var(--text-secondary);font-size:12px}.tile .w{color:var(--warn);font-size:12px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}
svg text{fill:var(--text-primary);font-size:12px}svg .muted{fill:var(--text-secondary)}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 10px;text-align:right;
 border-bottom:1px solid var(--grid)}th{color:var(--text-secondary);font-weight:500}td:first-child,th:first-child{text-align:left}
.gap{color:var(--warn)}ul.gaps li{margin:2px 0}
"""


def bar_chart(rows, width=640, label_w=150, row_h=26, target=None) -> str:
    """rows: [(라벨, 값, css색변수, 툴팁)] 가로 막대. target이 있으면 점선 기준선."""
    vmax = max([v for _, v, _, _ in rows] + ([target] if target else []) + [1])
    plot_w = width - label_w - 60
    h = row_h * len(rows) + 20
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for i, (lab, v, color, tip) in enumerate(rows):
        y = 10 + i * row_h
        w = max(2.0, plot_w * v / vmax)
        out.append(f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end">{html.escape(lab)}</text>')
        out.append(f'<g><title>{html.escape(tip)}</title><rect x="{label_w}" y="{y + 3}" width="{w:.1f}" '
                   f'height="{row_h - 8}" rx="4" fill="var({color})"/>'
                   f'<rect x="{label_w}" y="{y}" width="{plot_w}" height="{row_h}" fill="transparent"/></g>')
        out.append(f'<text x="{label_w + w + 6:.1f}" y="{y + 15}" class="muted">{v:,}</text>')
    if target:
        x = label_w + plot_w * target / vmax
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="4" y2="{h - 6}" stroke="var(--text-muted)" '
                   f'stroke-dasharray="4 3" stroke-width="1"/>'
                   f'<text x="{x + 4:.1f}" y="{h - 4}" class="muted">목표 {target}</text>')
    return "".join(out) + "</svg>"


def heat_table(row_labels, col_labels, values, flags=None) -> str:
    """순차 파랑 히트맵 표. 값은 항상 글자로 표시 (색만으로 읽지 않게). flags: 부족 칸 → ▼ 표시."""
    flat = [v for row in values for v in row]
    vmax = max(flat + [1])
    out = ['<div class="scroll"><table><tr><th></th>' + "".join(f"<th>{html.escape(c)}</th>" for c in col_labels) + "</tr>"]
    for r, (rl, row) in enumerate(zip(row_labels, values)):
        out.append(f"<tr><td>{html.escape(rl)}</td>")
        for c, v in enumerate(row):
            step = 0 if v == 0 else 1 + min(5, int(5 * v / vmax))
            fg = "#fff" if step >= 4 else "var(--text-primary)"
            mark = ' <span title="목표 미달">▼</span>' if flags and flags[r][c] else ""
            out.append(f'<td style="background:var(--seq-{step});color:{fg}" title="{html.escape(rl)} / '
                       f'{html.escape(col_labels[c])}: {v}">{v:,}{mark}</td>')
        out.append("</tr>")
    return "".join(out) + "</table></div>"


def render(res: dict, cfg: dict) -> str:
    t = cfg["targets"]
    objs = res["objects"]
    splits = res["splits"]
    split_name = {"train": "학습", "val": "검증", "test": "시험(평가 전용 맵)"}
    tiles = [(f"{split_name[s]} 이미지", f"{res['images'].get(s, 0):,}", "") for s in splits]
    for c in CLASS_ORDER:
        tot = sum(objs[c][s] for s in splits)
        warn = "목표 미달" if any(objs[c][s] < t[TOTAL_KEY[s]] for s in splits) else ""
        tiles.append((f'<span class="sw" style="background:var(--c-{c})"></span>{CLASS_LABEL[c]}', f"{tot:,}", warn))
    tile_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
                        f'{f"<div class=w>{w}</div>" if w else ""}</div>' for k, v, w in tiles)

    def split_rows(s):
        return [(f"{CLASS_LABEL[c]} ({s})", objs[c][s], f"--c-{c}", f"{CLASS_LABEL[c]} {s}: {objs[c][s]:,}")
                for c in CLASS_ORDER]

    conds = res["_conds"]
    pairs = [(c, s) for c in CLASS_ORDER for s in splits]
    cols = [f"{CLASS_LABEL[c]} {s}" for c, s in pairs]
    vals = [[res["_cond"].get(f"{cd}|{c}|{s}", 0) for c, s in pairs] for cd in conds]
    flags = [[v < t[CELL_KEY[s]] for v, (c, s) in zip(row, pairs)] for row in vals]

    tl_cls = [c for c in CLASS_ORDER if c.startswith("tl_")]
    size_vals = [[res["tl_size_hist"][c][b] for b in SIZE_LABELS] for c in tl_cls]
    ign = sorted(res["ignored"].items(), key=lambda kv: -kv[1])
    ign_rows = [(k, v, "--text-muted", f"{k}: {v:,}") for k, v in ign]

    gap_items = "".join(f'<li class="gap">{html.escape(g["text"])}</li>' for g in res["gaps"]) or "<li>없음</li>"
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DriveLoop 분포 리포트</title>
<style>{CSS}</style></head><body><div class="viz-root">
<h1>데이터셋 {html.escape(res['dataset'])} 분포 리포트</h1>
<p class="sub">중복 제거·무시 영역 처리 후 실제 학습에 들어가는 라벨 기준. 목표치: 클래스당 {' / '.join(f"{s} {t[TOTAL_KEY[s]]}" for s in splits)}.</p>
<div class="tiles">{tile_html}</div>

<h2>클래스 × 분할</h2><p class="sub">점선 = 목표치. 막대에 마우스를 올리면 수치.</p>
{"".join(f'<div class="scroll">{bar_chart(split_rows(s), target=t[TOTAL_KEY[s]])}</div>' for s in splits)}

<h2>조건(날씨·시간대)별 객체 수</h2><p class="sub">▼ = 칸 목표 미달 ({' / '.join(f"{s} {t[CELL_KEY[s]]}" for s in splits)} 미만)</p>
{heat_table(conds, cols, vals, flags)}

<h2>신호등 박스 높이(px) 분포</h2>
<p class="sub">{cfg['small_tl_px']}px 미만 비율 {res['tl_small_ratio'] * 100:.0f}% — 640 입력으로 줄이면 절반 크기가 된다.</p>
{heat_table([CLASS_LABEL[c] for c in tl_cls], SIZE_LABELS, size_vals)}

<h2>무시 영역(회색 처리) 사유</h2>
<div class="scroll">{bar_chart(ign_rows, label_w=200)}</div>

<h2>부족 목록 ({res['num_gaps']}건)</h2><ul class="gaps">{gap_items}</ul>
</div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v2")
    parser.add_argument("--config", default=str(CONFIG_DIR / "report.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name = args.dataset
    ds = PROJECT_ROOT / "data" / "datasets" / name
    out = PROJECT_ROOT / "data" / "reports" / name
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(ds / "manifest.parquet")
    export_report = json.loads((ds / "export_report.json").read_text(encoding="utf-8"))
    names = export_report["config"]["classes"]
    CLASS_ORDER[:] = [c for c in ALL_CLASSES if c in names]
    objs = load_objects(ds, manifest, names, img_h=720)
    res = analyze(objs, manifest, export_report, cfg, all_maps=sorted(manifest["map"].unique()))

    (out / "distribution.html").write_text(render(res, cfg), encoding="utf-8")
    metrics = {k: v for k, v in res.items() if not k.startswith("_")}
    (out / "distribution_report.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] → {out / 'distribution.html'}")
    print(f"         부족 {res['num_gaps']}건, 작은 신호등(<{cfg['small_tl_px']}px) {res['tl_small_ratio'] * 100:.0f}%")
    for g in res["gaps"]:
        if g["kind"] != "cell":
            print(f"         - {g['text']}")
    print(f"         - 조건 칸 부족 {sum(g['kind'] == 'cell' for g in res['gaps'])}건 (HTML 참고)")


if __name__ == "__main__":
    main()
