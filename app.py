import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="メラニンケア 色変化分析",
    page_icon="🔬",
    layout="wide",
)

STAGE_NAMES = [
    "ステージ1（施術前）",
    "ステージ2（1〜3ヶ月後）",
    "ステージ3（3〜6ヶ月後）",
]

# ── Image Processing ──────────────────────────────────────────────────────────

def load_bgr(uf):
    uf.seek(0)
    buf = np.frombuffer(uf.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def circle_mask(h, w, cx_pct, cy_pct, r_pct):
    cx = int(cx_pct / 100 * w)
    cy = int(cy_pct / 100 * h)
    r  = max(2, int(r_pct / 100 * min(h, w)))
    m  = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m

def mean_lab(img_bgr, mask):
    """Mean LAB color (OpenCV float32 scale) within mask."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px  = lab[mask > 0]
    return px.mean(axis=0) if len(px) else np.array([128.0, 128.0, 128.0])

def normalize(img_bgr, src_mean, tgt_mean):
    """Global LAB shift so src reference skin → tgt reference skin."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab += (tgt_mean - src_mean)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def darkness_score(img_bgr, mask):
    """
    Darkness = 100 - L*  (L* in 0–100).
    OpenCV L channel is 0–255, so L* = L_cv * 100/255.
    Higher number = darker / more pigmented.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px  = lab[:, :, 0][mask > 0]
    if not len(px):
        return 0.0
    return round(100.0 - float(px.mean()) * 100.0 / 255.0, 1)

def annotate(img_bgr, ref, tgt, max_w=500):
    """Draw green reference circle + red target circle, then resize."""
    h, w = img_bgr.shape[:2]
    out  = img_bgr.copy()
    for (cx_p, cy_p, r_p), color, label in [
        (ref, (0, 200, 0),   "REF"),
        (tgt, (30, 30, 220), "TARGET"),
    ]:
        cx = int(cx_p / 100 * w)
        cy = int(cy_p / 100 * h)
        r  = max(2, int(r_p / 100 * min(h, w)))
        cv2.circle(out, (cx, cy), r, color, 2)
        cv2.putText(out, label, (cx - r, max(cy - r - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    scale = min(max_w / w, 1.0)
    out   = cv2.resize(out, (int(w * scale), int(h * scale)))
    return to_rgb(out)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔬 メラニンケア 色変化分析アプリ")
st.caption("施術前後の写真をアップロードして、色素の濃さの変化を定量化します。")
st.markdown("---")

col_left, col_right = st.columns([3, 1])
with col_left:
    part_name = st.text_input("施術部位名", value="乳輪", help="例：乳輪、脇、膝、リップ など")
with col_right:
    n_stages = st.radio("ステージ数", [2, 3], horizontal=True)

st.markdown("---")

uploads    = {}  # {stage_idx: UploadedFile}
ref_params = {}  # {stage_idx: (cx%, cy%, r%)}
tgt_params = {}  # {stage_idx: (cx%, cy%, r%)}

for i in range(n_stages):
    sname = STAGE_NAMES[i]
    with st.expander(f"📷 {sname}", expanded=True):
        uf = st.file_uploader(
            f"{sname} の写真", type=["jpg", "jpeg", "png"], key=f"uf_{i}"
        )
        if uf:
            img = load_bgr(uf)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🟢 基準肌エリア**（施術していない正常な肌）")
                rx = st.slider("中心X %", 0, 100, 20, key=f"rx_{i}")
                ry = st.slider("中心Y %", 0, 100, 20, key=f"ry_{i}")
                rr = st.slider("半径 %",   1,  25,  5, key=f"rr_{i}",
                               help="画像の短辺に対する割合")
            with col2:
                st.markdown("**🔴 対象エリア**（施術部位）")
                tx = st.slider("中心X %", 0, 100, 50, key=f"tx_{i}")
                ty = st.slider("中心Y %", 0, 100, 55, key=f"ty_{i}")
                tr = st.slider("半径 %",   1,  25,  8, key=f"tr_{i}",
                               help="画像の短辺に対する割合")

            preview = annotate(img, (rx, ry, rr), (tx, ty, tr), max_w=640)
            st.image(preview,
                     caption="緑＝基準肌（REF） / 赤＝対象部位（TARGET）",
                     use_container_width=True)

            uploads[i]    = uf
            ref_params[i] = (rx, ry, rr)
            tgt_params[i] = (tx, ty, tr)
        else:
            st.info("写真をアップロードしてください。")

st.markdown("---")

ready = len(uploads) == n_stages
if not ready:
    st.caption(f"⬆ 全 {n_stages} ステージの写真をアップロードすると「分析実行」ボタンが有効になります。")

if st.button("🔬 分析を実行", type="primary", disabled=not ready) and ready:

    with st.spinner("画像を正規化して分析中…"):

        # Load all images fresh
        imgs = {i: load_bgr(uf) for i, uf in uploads.items()}

        # Stage 1 reference skin: defines the normalization target
        h0, w0   = imgs[0].shape[:2]
        m0       = circle_mask(h0, w0, *ref_params[0])
        ref0_lab = mean_lab(imgs[0], m0)

        # Normalize stage 2+ so their reference skin matches stage 1
        normed = {0: imgs[0].copy()}
        for i in range(1, n_stages):
            h, w   = imgs[i].shape[:2]
            mi     = circle_mask(h, w, *ref_params[i])
            normed[i] = normalize(imgs[i], mean_lab(imgs[i], mi), ref0_lab)

        # Darkness scores on normalized images
        scores = {}
        for i in range(n_stages):
            h, w      = normed[i].shape[:2]
            tm        = circle_mask(h, w, *tgt_params[i])
            scores[i] = darkness_score(normed[i], tm)

    st.success("✅ 分析完了！")

    # ── 1. Corrected images side by side ──────────────────────────────────────
    st.markdown("## 📷 補正後の画像比較")
    cols = st.columns(n_stages)
    for i, col in enumerate(cols):
        with col:
            ann = annotate(normed[i], ref_params[i], tgt_params[i], max_w=420)
            st.image(ann, caption=STAGE_NAMES[i])
            st.metric("色の濃さ", scores[i],
                      delta=round(scores[i] - scores[0], 1) if i > 0 else None,
                      delta_color="inverse")

    # ── 2. Before / After correction per stage ────────────────────────────────
    st.markdown("## 🔄 補正前後の比較")
    for i in range(n_stages):
        st.markdown(f"**{STAGE_NAMES[i]}**")
        c1, c2 = st.columns(2)
        with c1:
            st.image(annotate(imgs[i],   ref_params[i], tgt_params[i], max_w=420),
                     caption="補正前（元画像）")
        with c2:
            st.image(annotate(normed[i], ref_params[i], tgt_params[i], max_w=420),
                     caption="補正後（肌色正規化済み）")

    # ── 3. Bar chart ──────────────────────────────────────────────────────────
    st.markdown("## 📊 色の濃さ 推移グラフ")
    labels     = [STAGE_NAMES[i] for i in range(n_stages)]
    vals       = [scores[i] for i in range(n_stages)]
    bar_colors = ["#1e3c72", "#2a7dd4", "#38a3e8"][:n_stages]

    fig = go.Figure(go.Bar(
        x=labels,
        y=vals,
        marker_color=bar_colors,
        text=[str(v) for v in vals],
        textposition="outside",
        width=0.45,
    ))
    fig.update_layout(
        title=f"【{part_name}】色の濃さ変化（施術前 = {scores[0]}）",
        yaxis=dict(
            title="色の濃さ（0＝無色 〜 100＝最濃）",
            range=[0, 110],
        ),
        xaxis_title="ステージ",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        height=440,
        showlegend=False,
    )
    fig.add_hline(
        y=scores[0],
        line_dash="dot",
        line_color="#999",
        annotation_text=f"施術前ベース ({scores[0]})",
        annotation_position="top right",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 4. Change rate report ─────────────────────────────────────────────────
    st.markdown("## 📝 変化率レポート")
    base = scores[0]
    rows = [{
        "ステージ": STAGE_NAMES[0],
        "色の濃さ": base,
        "変化量": "—",
        "変化率（S1比）": "基準",
    }]

    for i in range(1, n_stages):
        d     = scores[i]
        delta = d - base
        rate  = (delta / base * 100) if base else 0
        icon  = "✅" if delta < 0 else "⚠️"
        word  = "薄くなりました" if delta < 0 else "濃くなりました"
        st.markdown(
            f"**{STAGE_NAMES[0]} → {STAGE_NAMES[i]}**  \n"
            f"{icon}  濃さ `{base}` → `{d}`　"
            f"変化量 **{delta:+.1f}**（**{abs(rate):.1f}%** {word}）"
        )
        rows.append({
            "ステージ": STAGE_NAMES[i],
            "色の濃さ": d,
            "変化量": f"{delta:+.1f}",
            "変化率（S1比）": f"{rate:+.1f}%",
        })

    st.markdown("### 集計表")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
