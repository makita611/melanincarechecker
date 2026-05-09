import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from PIL import Image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORD = True
except ImportError:
    HAS_COORD = False

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

SIZE_OPTIONS = {
    "小 (300px)": 300,
    "中 (500px)": 500,
    "大 (700px)": 700,
    "特大 (900px)": 900,
}


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
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px  = lab[mask > 0]
    return px.mean(axis=0) if len(px) else np.array([128.0, 128.0, 128.0])

def normalize(img_bgr, src_mean, tgt_mean):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab += (tgt_mean - src_mean)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def darkness_score(img_bgr, mask):
    """Darkness = 100 - L*  (L* in 0–100 scale)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px  = lab[:, :, 0][mask > 0]
    if not len(px):
        return 0.0
    return round(100.0 - float(px.mean()) * 100.0 / 255.0, 1)

def draw_circles(img_bgr, ref, tgt):
    """Draw green (REF) and red (TARGET) circles on a copy of img_bgr."""
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
        cv2.putText(out, label, (cx - r, max(cy - r - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def annotate_resized(img_bgr, ref, tgt, max_w=420):
    ann   = draw_circles(img_bgr, ref, tgt)
    h, w  = ann.shape[:2]
    scale = min(max_w / w, 1.0)
    ann   = cv2.resize(ann, (int(w * scale), int(h * scale)))
    return to_rgb(ann)


# ── Session state ──────────────────────────────────────────────────────────────

def init_stage(i):
    for k, v in [
        (f"rx_{i}", 20), (f"ry_{i}", 20), (f"rr_{i}", 5),
        (f"tx_{i}", 50), (f"ty_{i}", 55), (f"tr_{i}", 8),
        (f"last_click_{i}", None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🔬 メラニンケア 色変化分析アプリ")
st.caption("施術前後の写真をアップロードして、色素の濃さの変化を定量化します。")
st.markdown("---")

# Global settings
c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    part_name = st.text_input("施術部位名", value="乳輪",
                              help="例：乳輪、脇、膝、リップ、VIO など")
with c2:
    n_stages = st.radio("ステージ数", [2, 3], horizontal=True)
with c3:
    size_label = st.select_slider(
        "🖼 画像の表示サイズ",
        options=list(SIZE_OPTIONS.keys()),
        value="中 (500px)",
    )
    display_w = SIZE_OPTIONS[size_label]

st.markdown("---")

uploads    = {}
ref_params = {}
tgt_params = {}

for i in range(n_stages):
    init_stage(i)

    with st.expander(f"📷 {STAGE_NAMES[i]}", expanded=True):
        uf = st.file_uploader(
            f"{STAGE_NAMES[i]} の写真", type=["jpg", "jpeg", "png"], key=f"uf_{i}"
        )
        if uf:
            img            = load_bgr(uf)
            h_orig, w_orig = img.shape[:2]
            actual_w       = min(w_orig, display_w)
            actual_h       = int(h_orig * actual_w / w_orig)

            # Mode toggle — which circle does the next click set?
            mode = st.radio(
                "クリックで円の中心を設定",
                ["🟢 基準肌（REF）を設定中", "🔴 対象部位（TARGET）を設定中"],
                key=f"mode_{i}",
                horizontal=True,
                help="モードを選んで画像をクリックすると、その円の中心が移動します",
            )
            is_ref = "REF" in mode

            # Build annotated PIL image from current session state
            ref = (st.session_state[f"rx_{i}"],
                   st.session_state[f"ry_{i}"],
                   st.session_state[f"rr_{i}"])
            tgt = (st.session_state[f"tx_{i}"],
                   st.session_state[f"ty_{i}"],
                   st.session_state[f"tr_{i}"])
            pil = Image.fromarray(to_rgb(draw_circles(img, ref, tgt)))

            # Clickable image (or fallback static image)
            if HAS_COORD:
                coord = streamlit_image_coordinates(pil, key=f"coord_{i}", width=actual_w)
                if coord and coord != st.session_state[f"last_click_{i}"]:
                    st.session_state[f"last_click_{i}"] = coord
                    cx = min(100, max(0, int(coord["x"] / actual_w * 100)))
                    cy = min(100, max(0, int(coord["y"] / actual_h * 100)))
                    if is_ref:
                        st.session_state[f"rx_{i}"] = cx
                        st.session_state[f"ry_{i}"] = cy
                    else:
                        st.session_state[f"tx_{i}"] = cx
                        st.session_state[f"ty_{i}"] = cy
                    st.rerun()
            else:
                st.image(to_rgb(draw_circles(img, ref, tgt)), width=actual_w)
                st.warning("streamlit-image-coordinates が未インストールのためクリック選択は無効です。")

            st.caption("緑＝基準肌（REF） ／ 赤＝対象部位（TARGET）　　"
                       "※ スライダーで位置・半径を微調整できます")

            # Fine-tune sliders (bound to session state via key=)
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**🟢 基準肌（REF）の微調整**")
                st.slider("中心X %", 0, 100, key=f"rx_{i}")
                st.slider("中心Y %", 0, 100, key=f"ry_{i}")
                st.slider("半径 %",   1,  25, key=f"rr_{i}",
                          help="画像短辺に対する%")
            with cb:
                st.markdown("**🔴 対象部位（TARGET）の微調整**")
                st.slider("中心X %", 0, 100, key=f"tx_{i}")
                st.slider("中心Y %", 0, 100, key=f"ty_{i}")
                st.slider("半径 %",   1,  25, key=f"tr_{i}",
                          help="画像短辺に対する%")

            uploads[i]    = uf
            ref_params[i] = (st.session_state[f"rx_{i}"],
                             st.session_state[f"ry_{i}"],
                             st.session_state[f"rr_{i}"])
            tgt_params[i] = (st.session_state[f"tx_{i}"],
                             st.session_state[f"ty_{i}"],
                             st.session_state[f"tr_{i}"])
        else:
            st.info("写真をアップロードしてください。")

st.markdown("---")

ready = len(uploads) == n_stages
if not ready:
    st.caption(f"⬆ 全 {n_stages} ステージの写真をアップロードすると「分析実行」が有効になります。")

if st.button("🔬 分析を実行", type="primary", disabled=not ready) and ready:

    with st.spinner("画像を正規化して分析中…"):
        imgs = {i: load_bgr(uf) for i, uf in uploads.items()}

        # Stage 1 reference skin defines the normalization target
        h0, w0   = imgs[0].shape[:2]
        m0       = circle_mask(h0, w0, *ref_params[0])
        ref0_lab = mean_lab(imgs[0], m0)

        # Normalize stage 2+ to match stage 1 reference skin
        normed = {0: imgs[0].copy()}
        for i in range(1, n_stages):
            h, w      = imgs[i].shape[:2]
            mi        = circle_mask(h, w, *ref_params[i])
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
            st.image(annotate_resized(normed[i], ref_params[i], tgt_params[i]),
                     caption=STAGE_NAMES[i])
            st.metric(
                "色の濃さ",
                scores[i],
                delta=round(scores[i] - scores[0], 1) if i > 0 else None,
                delta_color="inverse",
            )

    # ── 2. Before / After correction per stage ────────────────────────────────
    st.markdown("## 🔄 補正前後の比較")
    for i in range(n_stages):
        st.markdown(f"**{STAGE_NAMES[i]}**")
        c1, c2 = st.columns(2)
        with c1:
            st.image(annotate_resized(imgs[i],   ref_params[i], tgt_params[i]),
                     caption="補正前（元画像）")
        with c2:
            st.image(annotate_resized(normed[i], ref_params[i], tgt_params[i]),
                     caption="補正後（肌色正規化済み）")

    # ── 3. Bar chart ──────────────────────────────────────────────────────────
    st.markdown("## 📊 色の濃さ 推移グラフ")
    labels     = [STAGE_NAMES[i] for i in range(n_stages)]
    vals       = [scores[i] for i in range(n_stages)]
    bar_colors = ["#1e3c72", "#2a7dd4", "#38a3e8"][:n_stages]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=bar_colors,
        text=[str(v) for v in vals],
        textposition="outside",
        width=0.45,
    ))
    fig.update_layout(
        title=f"【{part_name}】色の濃さ変化（施術前 = {scores[0]}）",
        yaxis=dict(title="色の濃さ（0＝無色 〜 100＝最濃）", range=[0, 110]),
        xaxis_title="ステージ",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        height=440,
        showlegend=False,
    )
    fig.add_hline(
        y=scores[0], line_dash="dot", line_color="#999",
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
