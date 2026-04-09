import streamlit as st
import joblib
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ – Claim Estimator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'DM Sans', sans-serif;
    background: #f0f4ff;
    color: #1a1f36;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e8edff 0%, #f5f7ff 60%, #eef1fb 100%);
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top Navigation Bar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2.5rem;
    background: #fff;
    border-bottom: 1px solid #e4e8f5;
    box-shadow: 0 2px 20px rgba(90,100,200,.06);
    margin-bottom: 2.5rem;
    border-radius: 0 0 18px 18px;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: .7rem;
    font-weight: 700;
    font-size: 1.25rem;
    color: #3c4fe0;
    letter-spacing: -.5px;
}
.navbar-brand .dot { color: #ff5c5c; }
.navbar-links {
    display: flex;
    gap: 2rem;
    font-size: .9rem;
    color: #6b7280;
    font-weight: 500;
}
.navbar-badge {
    background: #3c4fe0;
    color: #fff;
    border-radius: 20px;
    padding: .35rem 1rem;
    font-size: .8rem;
    font-weight: 600;
}

/* ── Page title ── */
.page-hero {
    text-align: center;
    margin-bottom: 2.5rem;
}
.page-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1f36;
    letter-spacing: -.8px;
    line-height: 1.2;
}
.page-hero p {
    color: #6b7280;
    font-size: 1.05rem;
    margin-top: .5rem;
    font-weight: 400;
}

/* ── Cards ── */
.card {
    background: #fff;
    border-radius: 22px;
    padding: 2rem 2.2rem;
    box-shadow: 0 4px 30px rgba(60,79,224,.07);
    border: 1px solid rgba(255,255,255,.8);
    margin-bottom: 1.4rem;
}
.card-title {
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9aa0b8;
    margin-bottom: 1.2rem;
}

/* ── Summary Panel ── */
.summary-card {
    background: linear-gradient(145deg, #3c4fe0 0%, #5b6df5 100%);
    border-radius: 22px;
    padding: 2rem;
    color: #fff;
    box-shadow: 0 12px 40px rgba(60,79,224,.28);
    text-align: center;
    height: 100%;
}
.summary-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,.2));
}
.summary-label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    opacity: .75;
    margin-bottom: .4rem;
}
.summary-amount {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1;
    margin-bottom: 1.5rem;
    font-family: 'DM Mono', monospace;
}
.summary-amount .currency {
    font-size: 1.4rem;
    vertical-align: super;
    opacity: .8;
}
.summary-divider {
    width: 50px;
    height: 2px;
    background: rgba(255,255,255,.3);
    margin: 0 auto 1.5rem;
    border-radius: 10px;
}
.summary-meta {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.summary-meta-item {
    background: rgba(255,255,255,.15);
    border-radius: 12px;
    padding: .9rem .8rem;
    backdrop-filter: blur(10px);
}
.summary-meta-item .meta-label {
    font-size: .65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: .7;
    display: block;
    margin-bottom: .3rem;
}
.summary-meta-item .meta-value {
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}

/* ── Step badges ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: #eef0ff;
    color: #3c4fe0;
    border-radius: 8px;
    padding: .35rem .8rem;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.step-num {
    background: #3c4fe0;
    color: #fff;
    border-radius: 4px;
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: .65rem;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #3c4fe0 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: #e4e8f5 !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label {
    font-weight: 600 !important;
    font-size: .85rem !important;
    color: #374151 !important;
    letter-spacing: .2px;
}
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #c4caf0 !important;
    background: #f8f9ff !important;
    color: #1a1f36 !important;
}

/* Selected value text */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span,
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] > div > div,
[data-baseweb="select"] input {
    color: #1a1f36 !important;
    font-weight: 500 !important;
}

/* Dropdown menu container */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
ul[role="listbox"] {
    background: #ffffff !important;
    border: 1px solid #dce0f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 30px rgba(60,79,224,.12) !important;
}

/* Dropdown menu options */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="option"] {
    background: #ffffff !important;
    color: #1a1f36 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
    padding: .65rem 1rem !important;
}

/* Hover state */
[role="option"]:hover,
[data-baseweb="menu"] li:hover {
    background: #eef0ff !important;
    color: #3c4fe0 !important;
}

/* Currently selected option highlight */
[aria-selected="true"] {
    background: #eef0ff !important;
    color: #3c4fe0 !important;
    font-weight: 600 !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3c4fe0 0%, #5b6df5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: .9rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: .3px !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    box-shadow: 0 8px 24px rgba(60,79,224,.3) !important;
    transition: all .2s ease !important;
    cursor: pointer !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(60,79,224,.42) !important;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
    border-radius: 18px;
    padding: 2rem;
    color: #fff;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 10px 40px rgba(26,31,54,.25);
}
.result-box .r-label {
    font-size: .72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: .6;
    margin-bottom: .6rem;
}
.result-box .r-amount {
    font-size: 2.6rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    letter-spacing: -1px;
    color: #a5b4fc;
}
.result-box .r-sub {
    font-size: .85rem;
    opacity: .5;
    margin-top: .4rem;
}

/* ── Progress dots ── */
.progress-dots {
    display: flex;
    justify-content: center;
    gap: .5rem;
    margin: 1.5rem 0 .5rem;
}
.dot-item {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #dce0f0;
}
.dot-item.active { background: #3c4fe0; width: 24px; border-radius: 4px; }

/* ── Info tag ── */
.info-tag {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: #fff7ed;
    color: #c2410c;
    border-radius: 8px;
    padding: .3rem .75rem;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .3px;
    margin-bottom: 1.2rem;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #dce0f0, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('Random_Forest_Model.pkl', 'rb') as f:
            return joblib.load(f)
    except Exception:
        return None


data = load_model()

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">🛡️ InsureIQ<span class="dot">.</span></div>
  <div class="navbar-links">
    <span>Dashboard</span>
    <span>Claims</span>
    <span>Reports</span>
  </div>
  <div class="navbar-badge">AI-Powered</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>Estimate Your Insurance Claim</h1>
  <p>Fill in the details below and get an instant AI-powered claim prediction.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: left form | right summary ────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Step 1: Personal Info ──────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge"><span class="step-num">1</span>Personal Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Claimant Information</div>', unsafe_allow_html=True)

    age = st.slider("Claimant Age", min_value=18, max_value=70, value=35, step=1,
                    help="Age of the primary insurance claimant")

    col_a, col_b = st.columns(2)
    with col_a:
        incident_state = st.selectbox("State of Incident",
                                      ['NC', 'NY', 'OH', 'PA', 'SC', 'VA', 'WV'])
    with col_b:
        authorities = st.selectbox("Authorities Contacted",
                                   ['Ambulance', 'Fire', 'None', 'Other', 'Police'])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 2: Vehicle Info ───────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge"><span class="step-num">2</span>Vehicle Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Car & Collision Information</div>', unsafe_allow_html=True)

    auto_model_options = [
        '93','95','3 Series','92x','A3','A5','Accord','C300','Camry','Civic',
        'Corolla','CRV','E400','Escape','F150','Forrestor','Fusion','Grand Cherokee',
        'Highlander','Impreza','Jetta','Legacy','M5','Malibu','Maxima','MDX','ML350',
        'Neon','Passat','Pathfinder','RAM','RSX','Silverado','Tahoe','TL','Ultima',
        'Wrangler','X5','X6'
    ]
    auto_model = st.selectbox("Car Model", auto_model_options)

    col_c, col_d = st.columns(2)
    with col_c:
        collision_type = st.selectbox("Type of Collision",
                                      ['Not answered', 'Front Collision', 'Rear Collision', 'Side Collision'])
    with col_d:
        incident_type = st.selectbox("Type of Incident",
                                     ['Multi-vehicle Collision', 'Parked Car',
                                      'Single Vehicle Collision', 'Vehicle Theft'])

    vehicles_involved = st.slider("Vehicles Involved", min_value=1, max_value=5, value=1, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 3: Incident Info ──────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge"><span class="step-num">3</span>Incident Severity</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Damage Assessment</div>', unsafe_allow_html=True)

    incident_severity = st.selectbox("Severity of Incident",
                                     ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Claim Amount", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Right: Summary Panel ───────────────────────────────────────────────────
with right:
    st.markdown("""
    <div class="summary-card">
      <span class="summary-icon">🛡️</span>
      <div class="summary-label">Claim Estimator</div>
      <div class="summary-amount"><span class="currency">₦</span>—</div>
      <div class="summary-divider"></div>
      <div style="font-size:.85rem;opacity:.75;line-height:1.6;">
        Complete the form on the left and click <strong>Predict</strong> to get your 
        AI-powered claim estimate instantly.
      </div>
      <div class="summary-meta">
        <div class="summary-meta-item">
          <span class="meta-label">Model</span>
          <span class="meta-value">Random Forest</span>
        </div>
        <div class="summary-meta-item">
          <span class="meta-label">Features</span>
          <span class="meta-value">8 Inputs</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Result box appears after prediction
    if predict_btn:
        if data is None:
            st.error("⚠️ Model file not found. Place `Random_Forest_Model.pkl` in the app directory.")
        else:
            try:
                model         = data["model"]
                auth_enc      = data["authorities_contacted_encoder"]
                auto_enc      = data["auto_model_encoder"]
                coll_enc      = data["collision_type_encoder"]
                sev_enc       = data["incident_severity_encoder"]
                state_enc     = data["incident_state_encoder"]
                itype_enc     = data["incident_type_encoder"]

                X = np.array([[age, authorities, auto_model, collision_type,
                               incident_severity, incident_state, incident_type,
                               vehicles_involved]], dtype=object)

                X[:, 1] = auth_enc.transform(X[:, 1])
                X[:, 2] = auto_enc.transform(X[:, 2])
                X[:, 3] = coll_enc.transform(X[:, 3])
                X[:, 4] = sev_enc.transform(X[:, 4])
                X[:, 5] = state_enc.transform(X[:, 5])
                X[:, 6] = itype_enc.transform(X[:, 6])

                amount = model.predict(X.astype(float))[0]

                st.markdown(f"""
                <div class="result-box">
                  <div class="r-label">Estimated Claim Amount</div>
                  <div class="r-amount">₦{amount:,.2f}</div>
                  <div class="r-sub">Based on {vehicles_involved} vehicle(s) · {incident_severity}</div>
                </div>
                """, unsafe_allow_html=True)

                # Quick insight chips
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-top:1rem;">
                  <div style="background:#fff;border-radius:14px;padding:1rem;border:1px solid #e4e8f5;text-align:center;">
                    <div style="font-size:.65rem;letter-spacing:1px;text-transform:uppercase;color:#9aa0b8;margin-bottom:.3rem;">Incident Type</div>
                    <div style="font-weight:700;font-size:.9rem;color:#1a1f36;">{incident_type}</div>
                  </div>
                  <div style="background:#fff;border-radius:14px;padding:1rem;border:1px solid #e4e8f5;text-align:center;">
                    <div style="font-size:.65rem;letter-spacing:1px;text-transform:uppercase;color:#9aa0b8;margin-bottom:.3rem;">Collision Type</div>
                    <div style="font-weight:700;font-size:.9rem;color:#1a1f36;">{collision_type}</div>
                  </div>
                  <div style="background:#fff;border-radius:14px;padding:1rem;border:1px solid #e4e8f5;text-align:center;">
                    <div style="font-size:.65rem;letter-spacing:1px;text-transform:uppercase;color:#9aa0b8;margin-bottom:.3rem;">State</div>
                    <div style="font-weight:700;font-size:.9rem;color:#1a1f36;">{incident_state}</div>
                  </div>
                  <div style="background:#fff;border-radius:14px;padding:1rem;border:1px solid #e4e8f5;text-align:center;">
                    <div style="font-size:.65rem;letter-spacing:1px;text-transform:uppercase;color:#9aa0b8;margin-bottom:.3rem;">Authorities</div>
                    <div style="font-weight:700;font-size:.9rem;color:#1a1f36;">{authorities}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#9aa0b8;font-size:.8rem;">
  InsureIQ &nbsp;·&nbsp; AI-Powered Claims Estimation &nbsp;·&nbsp; 
  <span style="color:#3c4fe0;font-weight:600;">Powered by Random Forest</span>
</div>
""", unsafe_allow_html=True)
