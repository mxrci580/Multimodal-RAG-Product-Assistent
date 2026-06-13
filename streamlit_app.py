import streamlit as st
from app.multimodal_agent import MultimodalAgent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Force light mode always */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #F8F9FA !important;
    color: #111827 !important;
}

/* Reset dark theme overrides */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #111827 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] * { color: #111827 !important; }

.sidebar-logo {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111827 !important;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}
.sidebar-sub {
    font-size: 0.75rem;
    color: #9CA3AF !important;
    margin-bottom: 1.2rem;
}

.step-row {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    margin-bottom: 1.1rem;
}
.step-dot {
    min-width: 20px; height: 20px;
    background: #111827;
    color: #fff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-weight: 700;
    margin-top: 1px; flex-shrink: 0;
}
.step-text h5 {
    margin: 0 0 1px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #111827 !important;
}
.step-text p {
    margin: 0;
    font-size: 0.73rem;
    color: #6B7280 !important;
    line-height: 1.4;
}

.stack-pill {
    display: inline-block;
    background: #F3F4F6;
    color: #374151 !important;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 2px 2px 2px 0;
}

/* ── Hero ── */
.hero-wrap {
    background: #ffffff;
    border-bottom: 1px solid #E5E7EB;
    padding: 2rem 0 1.5rem;
    margin-bottom: 1.5rem;
}
.hero-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6B7280;
    margin: 0 0 0.45rem;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6B7280;
    margin: 0;
}

/* ── Field label ── */
.field-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Recommendation box ── */
.rec-wrap {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.rec-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #111827;
    border-radius: 12px 0 0 12px;
}
.rec-eyebrow {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 0 0 0.6rem;
}
.rec-text {
    font-size: 0.92rem;
    color: #1F2937;
    line-height: 1.7;
    margin: 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 0 0 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E5E7EB;
}

/* ── Product card ── */
.pcard {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.pcard:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}
.pcard-img-wrap {
    width: 100%;
    aspect-ratio: 4/3;
    overflow: hidden;
    background: #F9FAFB;
    display: flex;
    align-items: center;
    justify-content: center;
}
.pcard-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
.pcard-img-placeholder {
    width: 100%;
    aspect-ratio: 4/3;
    background: #F3F4F6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: #D1D5DB;
}
.pcard-body {
    padding: 0.85rem 1rem 1rem;
    border-top: 1px solid #F3F4F6;
}
.pcard-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.4;
    margin: 0 0 0.6rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.pcard-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.pcard-price {
    font-size: 0.95rem;
    font-weight: 700;
    color: #111827;
}
.pcard-rating {
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 0.75rem;
    color: #D97706;
    font-weight: 600;
}

/* ── Empty state ── */
.empty-state {
    background: #ffffff;
    border: 1.5px dashed #E5E7EB;
    border-radius: 12px;
    height: 360px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
}
.empty-icon {
    width: 48px; height: 48px;
    background: #F3F4F6;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}
.empty-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    margin: 0 0 0.3rem;
}
.empty-sub {
    font-size: 0.8rem;
    color: #9CA3AF;
    margin: 0;
    line-height: 1.5;
}

/* ── Streamlit button override ── */
[data-testid="stButton"] > button {
    background: #111827 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.2rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.82 !important;
}
[data-testid="stButton"] > button:disabled {
    background: #D1D5DB !important;
    color: #9CA3AF !important;
    cursor: not-allowed !important;
}

/* ── Upload-ready state ── */
.upload-hint {
    font-size: 0.72rem;
    color: #9CA3AF;
    text-align: center;
    margin-top: 0.4rem;
}

/* ── Awaiting-click state (right col) ── */
.await-state {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.5rem;
}
.await-state p {
    font-size: 0.85rem;
    color: #6B7280;
    margin: 0;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1.5px dashed #D1D5DB !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #374151 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #6B7280 !important; }

/* ── Info box ── */
[data-testid="stAlert"] {
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    color: #6B7280 !important;
    border-radius: 8px !important;
}

/* ── Image caption ── */
[data-testid="caption"] {
    color: #9CA3AF !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def star_display(rating_str):
    try:
        val = float(str(rating_str).replace("★", "").strip())
        filled = int(val)
        half   = val - filled >= 0.5
        empty  = 5 - filled - (1 if half else 0)
        return "★" * filled + ("½" if half else "") + "☆" * empty, val
    except Exception:
        return "★★★★☆", 4.0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛍️ MPRS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Multimodal Product Recommendation System</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**How it works**")
    steps = [
        ("Upload image",    "Drop a photo of any product."),
        ("Vision analysis", "Gemini Vision describes its key attributes."),
        ("Vector search",   "Description is embedded and matched in ChromaDB."),
        ("RAG generation",  "Top matches are fed to Gemini for a tailored answer."),
        ("Results",         "AI recommendation + closest products appear."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f"""<div class="step-row">
                  <div class="step-dot">{i}</div>
                  <div class="step-text"><h5>{title}</h5><p>{desc}</p></div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Tech stack**", unsafe_allow_html=True)
    for tag in ["Gemini Vision", "Gemini API", "ChromaDB", "RAG", "Streamlit", "Docker"]:
        st.markdown(f'<span class="stack-pill">{tag}</span>', unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Gemini Vision · ChromaDB · RAG · Streamlit</div>
  <div class="hero-title">Multimodal Product Recommendation System</div>
  <div class="hero-sub">Upload any product image — the AI pipeline analyzes it and surfaces the closest matching products instantly.</div>
</div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.65], gap="large")

# ── LEFT col ─────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<span class="field-label">Product image</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload product image",
        type=["png", "jpg", "jpeg"],
        key="product_image",
        label_visibility="collapsed",
    )

    image_path = "temp_image.png"

    if uploaded_file:
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, use_container_width=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Button always rendered — disabled until a file is uploaded
    run = st.button(
        "🔍  Analyze & Find Similar Products",
        key="find_products",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    if not uploaded_file:
        st.markdown(
            '<div class="upload-hint">Upload an image above to enable search</div>',
            unsafe_allow_html=True,
        )


# ── RIGHT col ─────────────────────────────────────────────────────────────────
with col_right:

    if uploaded_file and run:

        # ── Run agent ──
        with st.spinner("Analyzing image with Gemini Vision…"):
            agent    = MultimodalAgent()
            response = agent.run(image_path)

        answer   = response.get("answer", "")
        products = response.get("products", [])

        # ── AI Recommendation ──
        st.markdown(
            f"""<div class="rec-wrap">
                  <div class="rec-eyebrow">AI Recommendation</div>
                  <div class="rec-text">{answer}</div>
                </div>""",
            unsafe_allow_html=True,
        )

        # ── Product cards ──
        if products:
            count = len(products)
            st.markdown(
                f'<div class="section-label">Top {count} matching product{"s" if count != 1 else ""}</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(2, gap="medium")
            for idx, product in enumerate(products):
                name   = product.get("product_name", "Unknown Product")
                price  = product.get("price", "—")
                rating = product.get("rating", "")
                img_url = product.get("image_url", "")
                stars_str, rating_val = star_display(rating)

                with cols[idx % 2]:
                    # Image area
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        st.markdown(
                            '<div class="pcard-img-placeholder">🛍️</div>',
                            unsafe_allow_html=True,
                        )

                    # Info area
                    st.markdown(
                        f"""<div class="pcard-body" style="background:#fff;border:1px solid #E5E7EB;
                                border-top:none;border-radius:0 0 12px 12px;
                                padding:0.8rem 0.9rem 0.9rem;">
                              <div class="pcard-name">{name}</div>
                              <div class="pcard-footer">
                                <span class="pcard-price">💰 {price}</span>
                                <span class="pcard-rating">{stars_str}&nbsp;{rating_val}</span>
                              </div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='margin-bottom:0.25rem'></div>", unsafe_allow_html=True)

        else:
            st.warning("No matching products returned. Try uploading a different image.")

    elif uploaded_file and not run:
        # File uploaded, button not yet clicked
        st.markdown("""
        <div class="await-state">
          <div style="font-size:1.8rem">🔍</div>
          <p><strong>Image ready.</strong></p>
          <p>Click <strong>Analyze &amp; Find Similar Products</strong> to run the AI pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🛒</div>
          <div class="empty-title">Results will appear here</div>
          <div class="empty-sub">Upload a product image on the left,<br>then click <strong>Analyze &amp; Find Similar Products</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
