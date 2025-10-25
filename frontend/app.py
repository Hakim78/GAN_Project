import streamlit as st
from pathlib import Path
from fe_utils.lottie import render_lottie
import time

st.set_page_config(page_title="Studio Génératif Pro", page_icon="🎬", layout="wide")

# CSS amélioré avec Bento Box Design
st.markdown("""
<style>
/* Variables globales */
:root {
    --primary: #8A2BE2;
    --secondary: #00E5FF;
    --accent: #FF6B6B;
    --bg-dark: #0B0F14;
    --bg-card: #1A1F2E;
    --bg-hover: #252B3B;
    --text-primary: #FFFFFF;
    --text-secondary: #A0AEC0;
    --border: rgba(138, 43, 226, 0.2);
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

/* Reset Streamlit */
.stApp {
    background: var(--bg-dark) !important;
}

/* Container principal */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* Hero Section avec gradient animé */
.hero-section {
    background: linear-gradient(135deg, rgba(138,43,226,0.1) 0%, rgba(0,229,255,0.1) 100%);
    border-radius: 24px;
    padding: 3rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(138,43,226,0.05) 0%, transparent 70%);
    animation: pulse 8s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

/* Bento Box Grid */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.bento-card {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid var(--border);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.bento-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.bento-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow);
    border-color: var(--primary);
    background: var(--bg-hover);
}

.bento-card:hover::before {
    transform: scaleX(1);
}

/* Cards avec icônes */
.feature-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(26, 31, 46, 0.8) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.feature-card:hover {
    border-color: var(--secondary);
    box-shadow: 0 4px 20px rgba(0, 229, 255, 0.2);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 10px currentColor);
}

/* Métriques style Revolut */
.metric-card {
    background: linear-gradient(135deg, rgba(138,43,226,0.15) 0%, rgba(0,229,255,0.15) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 24px rgba(138, 43, 226, 0.3);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Boutons style Netflix */
.custom-button {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 1rem 2rem;
    border-radius: 12px;
    border: none;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(138, 43, 226, 0.4);
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(138, 43, 226, 0.6);
}

/* Timeline style moderne */
.timeline {
    position: relative;
    padding: 2rem 0;
}

.timeline-item {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid var(--primary);
    transition: all 0.3s ease;
}

.timeline-item:hover {
    border-left-width: 8px;
    transform: translateX(8px);
}

/* Tags et badges */
.tag {
    display: inline-block;
    background: rgba(138, 43, 226, 0.2);
    color: var(--primary);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.25rem;
    border: 1px solid var(--primary);
}

/* Progress bar animée */
.progress-bar {
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 10px;
    animation: progress 2s ease-in-out infinite;
}

@keyframes progress {
    0% { width: 0%; }
    50% { width: 70%; }
    100% { width: 100%; }
}

/* Scroll effects */
.scroll-section {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.6s ease forwards;
}

@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(26, 31, 46, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* Hover effects sur images */
.image-hover {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
}

.image-hover img {
    transition: transform 0.5s ease;
}

.image-hover:hover img {
    transform: scale(1.1);
}

/* Typography améliorée */
h1 {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

h2 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2rem;
}

h3 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 1rem 0;
}

p {
    color: var(--text-secondary);
    line-height: 1.8;
    font-size: 1.05rem;
}

/* Responsive */
@media (max-width: 768px) {
    .bento-grid {
        grid-template-columns: 1fr;
    }
    
    h1 {
        font-size: 2.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section scroll-section">
    <h1>Studio Génératif IA</h1>
    <p style="font-size: 1.3rem; color: var(--text-secondary); max-width: 800px;">
        Je génère des visages photoréalistes avec trois architectures de pointe : 
        GANs classiques, Attention CBAM, et Diffusion Models.
    </p>
</div>
""", unsafe_allow_html=True)

# Métriques principales (style Revolut)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="feature-icon">🎨</div>
        <div class="metric-value">200K+</div>
        <div class="metric-label">Images Dataset</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="feature-icon">⚡</div>
        <div class="metric-value">2.3s</div>
        <div class="metric-label">Génération</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="feature-icon">📊</div>
        <div class="metric-value">45.8</div>
        <div class="metric-label">FID Score</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="feature-icon">🚀</div>
        <div class="metric-value">3</div>
        <div class="metric-label">Architectures</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Bento Box Grid - Modèles
st.markdown("""
<h2 style="text-align: center; margin: 3rem 0 2rem;">Mes Architectures</h2>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown("""
    <div class="bento-card scroll-section">
        <div class="feature-icon">🔷</div>
        <h3>V1 - Baseline</h3>
        <p>GAN classique avec architecture Conv2DTranspose progressive. 
        Rapide et efficace pour prototypage.</p>
        <div class="tag">Conv2D</div>
        <div class="tag">BatchNorm</div>
        <div class="tag">52.3 FID</div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="bento-card scroll-section" style="animation-delay: 0.1s;">
        <div class="feature-icon" style="color: var(--secondary);">💎</div>
        <h3>V2 - Attention</h3>
        <p>Architecture enrichie avec mécanismes CBAM (Channel + Spatial Attention). 
        Meilleure cohérence globale.</p>
        <div class="tag">CBAM</div>
        <div class="tag">Attention</div>
        <div class="tag">45.8 FID</div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="bento-card scroll-section" style="animation-delay: 0.2s;">
        <div class="feature-icon" style="color: var(--accent);">🌟</div>
        <h3>V3 - Diffusion</h3>
        <p>Modèle de diffusion (DDPM) avec génération conditionnelle. 
        Qualité maximale, contrôle par attributs.</p>
        <div class="tag">U-Net</div>
        <div class="tag">DDPM</div>
        <div class="tag">38.2 FID</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section Features avec Lottie
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="glass-card scroll-section">
        <h2>Fonctionnalités</h2>
        <div class="timeline">
            <div class="timeline-item">
                <h3>⚡ Génération instantanée</h3>
                <p>Je génère 16 visages en moins de 2 secondes avec V1/V2</p>
            </div>
            <div class="timeline-item">
                <h3>🎯 Comparaison objective</h3>
                <p>J'évalue mes modèles avec FID scores et métriques de diversité</p>
            </div>
            <div class="timeline-item">
                <h3>🎨 Génération conditionnelle</h3>
                <p>Je contrôle les attributs avec V3 (genre, sourire, lunettes...)</p>
            </div>
            <div class="timeline-item">
                <h3>📊 Interface interactive</h3>
                <p>Je visualise et télécharge mes résultats en temps réel</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    render_lottie("lotties/home.json", height=400)

st.markdown("<br><br>", unsafe_allow_html=True)

# Call-to-action buttons (style Netflix)
st.markdown("""
<h2 style="text-align: center; margin: 3rem 0 2rem;">Commencer maintenant</h2>
""", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button("🚀 V1 Baseline", use_container_width=True, type="primary"):
        st.switch_page("pages/1_V1_GAN_Baseline.py")

with col_btn2:
    if st.button("⚡ V2 Attention", use_container_width=True):
        st.switch_page("pages/2_V2_Attention_GAN.py")

with col_btn3:
    if st.button("🌟 V3 Diffusion", use_container_width=True):
        st.switch_page("pages/3_V3_Diffusion.py")

with col_btn4:
    if st.button("🔬 Comparer", use_container_width=True):
        st.switch_page("pages/4_Compare.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer avec stats
st.markdown("""
<div class="glass-card" style="text-align: center; padding: 2rem; margin-top: 4rem;">
    <h3 style="margin-bottom: 1rem;">Projet Deep Learning - GANs & Diffusion</h3>
    <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
        Dataset CelebA • TensorFlow 2.15 • Streamlit • 600+ epochs d'entraînement
    </p>
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
        <span class="tag">Python</span>
        <span class="tag">TensorFlow</span>
        <span class="tag">Keras</span>
        <span class="tag">Computer Vision</span>
        <span class="tag">Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)