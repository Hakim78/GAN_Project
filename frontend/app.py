import streamlit as st
from pathlib import Path
from fe_utils.lottie import render_lottie
import base64

st.set_page_config(
    page_title="Studio Génératif Pro", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Je charge le CSS externe
css_file = Path(__file__).parent / "ui" / "styles.css"
with open(css_file, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Fonction pour encoder la vidéo en base64
def get_video_base64(video_path):
    """Je convertis la vidéo en base64 pour l'afficher en background"""
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode()


# Fonction pour encoder les images en base64
def get_image_base64(image_path):
    """Je convertis l'image en base64"""
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    return base64.b64encode(img_bytes).decode()


# Je charge les assets
video_path = Path(__file__).parent / "assets" / "video" / "home_animation.mp4"
video_base64 = get_video_base64(video_path)

# Je charge les icônes
icons = {
    'dataset': get_image_base64(Path(__file__).parent / "assets" / "img" / "icons_dataset.png"),
    'generation': get_image_base64(Path(__file__).parent / "assets" / "img" / "icons_generation.png"),
    'fid': get_image_base64(Path(__file__).parent / "assets" / "img" / "icons_fid_score.png"),
    'arch': get_image_base64(Path(__file__).parent / "assets" / "img" / "icons_architectures.png")
}

# Je charge les backgrounds des versions
backgrounds = {
    'v1': get_image_base64(Path(__file__).parent / "assets" / "img" / "v1_background_img.png"),
    'v2': get_image_base64(Path(__file__).parent / "assets" / "img" / "v2_background_img.png"),
    'v3': get_image_base64(Path(__file__).parent / "assets" / "img" / "v3_background_img.png")
}


# Hero Section avec vidéo en background
st.markdown(f"""
<div class="hero-video-container">
    <video class="hero-video" autoplay muted loop playsinline>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <h1 class="hero-title">Studio Génératif IA</h1>
        <p class="hero-subtitle">
            Je génère des visages photoréalistes avec trois architectures de pointe
        </p>
        <div class="hero-tags">
            <span class="hero-tag">GANs Classiques</span>
            <span class="hero-tag">Attention CBAM</span>
            <span class="hero-tag">Diffusion Models</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Métriques principales avec glassmorphism
st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-glass-card">
        <div class="metric-icon-container">
            <img src="data:image/png;base64,{icons['dataset']}" class="metric-icon" alt="Dataset">
        </div>
        <div class="metric-content">
            <div class="metric-value">200K+</div>
            <div class="metric-label">Images Dataset</div>
        </div>
        <div class="metric-gradient"></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-glass-card">
        <div class="metric-icon-container">
            <img src="data:image/png;base64,{icons['generation']}" class="metric-icon" alt="Génération">
        </div>
        <div class="metric-content">
            <div class="metric-value">2.3s</div>
            <div class="metric-label">Génération</div>
        </div>
        <div class="metric-gradient"></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-glass-card">
        <div class="metric-icon-container">
            <img src="data:image/png;base64,{icons['fid']}" class="metric-icon" alt="FID Score">
        </div>
        <div class="metric-content">
            <div class="metric-value">45.8</div>
            <div class="metric-label">FID Score</div>
        </div>
        <div class="metric-gradient"></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-glass-card">
        <div class="metric-icon-container">
            <img src="data:image/png;base64,{icons['arch']}" class="metric-icon" alt="Architectures">
        </div>
        <div class="metric-content">
            <div class="metric-value">3</div>
            <div class="metric-label">Architectures</div>
        </div>
        <div class="metric-gradient"></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section titre architectures
st.markdown("""
<div class="section-header">
    <h2 class="section-title">Mes Architectures</h2>
    <p class="section-subtitle">Trois approches, une seule mission : générer des visages réalistes</p>
</div>
""", unsafe_allow_html=True)

# Bento Box Grid avec images de background
col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown(f"""
    <div class="model-card" data-version="v1">
        <div class="model-background" style="background-image: url('data:image/png;base64,{backgrounds['v1']}');"></div>
        <div class="model-overlay"></div>
        <div class="model-content">
            <div class="model-badge">Version 1</div>
            <h3 class="model-title">GAN Baseline</h3>
            <p class="model-description">
                Architecture classique avec Conv2DTranspose progressive. 
                Rapide et efficace pour prototypage.
            </p>
            <div class="model-tags">
                <span class="model-tag">Conv2D</span>
                <span class="model-tag">BatchNorm</span>
                <span class="model-tag">FID: 52.3</span>
            </div>
            <div class="model-stats">
                <div class="stat-item">
                    <span class="stat-value">1.2s</span>
                    <span class="stat-label">Génération</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">2.5M</span>
                    <span class="stat-label">Paramètres</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="model-card" data-version="v2">
        <div class="model-background" style="background-image: url('data:image/png;base64,{backgrounds['v2']}');"></div>
        <div class="model-overlay"></div>
        <div class="model-content">
            <div class="model-badge model-badge-cyan">Version 2</div>
            <h3 class="model-title">GAN + Attention</h3>
            <p class="model-description">
                Architecture enrichie avec CBAM (Channel + Spatial Attention). 
                Meilleure cohérence globale des visages.
            </p>
            <div class="model-tags">
                <span class="model-tag">CBAM</span>
                <span class="model-tag">Attention</span>
                <span class="model-tag">FID: 45.8</span>
            </div>
            <div class="model-stats">
                <div class="stat-item">
                    <span class="stat-value">1.5s</span>
                    <span class="stat-label">Génération</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">2.8M</span>
                    <span class="stat-label">Paramètres</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <div class="model-card" data-version="v3">
        <div class="model-background" style="background-image: url('data:image/png;base64,{backgrounds['v3']}');"></div>
        <div class="model-overlay"></div>
        <div class="model-content">
            <div class="model-badge model-badge-accent">Version 3</div>
            <h3 class="model-title">Diffusion Model</h3>
            <p class="model-description">
                DDPM avec génération conditionnelle par attributs. 
                Qualité maximale et contrôle précis.
            </p>
            <div class="model-tags">
                <span class="model-tag">U-Net</span>
                <span class="model-tag">DDPM</span>
                <span class="model-tag">FID: 38.2</span>
            </div>
            <div class="model-stats">
                <div class="stat-item">
                    <span class="stat-value">23s</span>
                    <span class="stat-label">Génération</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">5.2M</span>
                    <span class="stat-label">Paramètres</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section Features
st.markdown("""
<div class="features-section">
    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-number">01</div>
            <h3 class="feature-title">Génération Instantanée</h3>
            <p class="feature-text">Je génère 16 visages en moins de 2 secondes avec V1/V2</p>
        </div>
        <div class="feature-item">
            <div class="feature-number">02</div>
            <h3 class="feature-title">Comparaison Objective</h3>
            <p class="feature-text">J'évalue mes modèles avec FID scores et métriques de diversité</p>
        </div>
        <div class="feature-item">
            <div class="feature-number">03</div>
            <h3 class="feature-title">Génération Conditionnelle</h3>
            <p class="feature-text">Je contrôle les attributs avec V3 (genre, sourire, lunettes...)</p>
        </div>
        <div class="feature-item">
            <div class="feature-number">04</div>
            <h3 class="feature-title">Interface Interactive</h3>
            <p class="feature-text">Je visualise et télécharge mes résultats en temps réel</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# CTA Section
st.markdown("""
<div class="cta-section">
    <h2 class="cta-title">Prêt à générer ?</h2>
    <p class="cta-subtitle">Choisissez votre architecture et commencez la génération</p>
</div>
""", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button("V1 Baseline", use_container_width=True, type="primary", key="btn_v1"):
        st.switch_page("pages/1_V1_GAN_Baseline.py")

with col_btn2:
    if st.button("V2 Attention", use_container_width=True, key="btn_v2"):
        st.switch_page("pages/2_V2_Attention_GAN.py")

with col_btn3:
    if st.button("V3 Diffusion", use_container_width=True, key="btn_v3"):
        st.switch_page("pages/3_V3_Diffusion.py")

with col_btn4:
    if st.button("Comparer", use_container_width=True, key="btn_compare"):
        st.switch_page("pages/4_Compare.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-section">
    <div class="footer-content">
        <h3 class="footer-title">Projet Deep Learning - GANs & Diffusion</h3>
        <p class="footer-text">
            Dataset CelebA • TensorFlow 2.15 • Streamlit • 600+ epochs d'entraînement
        </p>
        <div class="footer-tags">
            <span class="footer-tag">Python</span>
            <span class="footer-tag">TensorFlow</span>
            <span class="footer-tag">Keras</span>
            <span class="footer-tag">Computer Vision</span>
            <span class="footer-tag">Deep Learning</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)