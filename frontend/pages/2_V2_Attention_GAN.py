import streamlit as st
from pathlib import Path
from fe_utils.loader import load_gan_generator, sample_gan, get_generator_z_dim
from fe_utils.preview import make_grid
import base64

st.set_page_config(page_title="V2 – Attention GAN", page_icon="💎", layout="wide")

# Je charge le CSS
css_file = Path(__file__).parents[1] / "ui" / "styles.css"
with open(css_file, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Je charge l'image de background V2
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_v2 = get_image_base64(Path(__file__).parents[1] / "assets" / "img" / "v2_background_img.png")

# Hero avec background image
st.markdown(f"""
<div class="page-hero" style="background-image: url('data:image/png;base64,{bg_v2}');">
    <div class="page-hero-overlay"></div>
    <div class="page-hero-content">
        <div class="page-badge page-badge-cyan">Version 2</div>
        <h1 class="page-title">GAN avec Attention CBAM</h1>
        <p class="page-subtitle">Mécanismes d'attention pour une meilleure cohérence</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section description
col_intro1, col_intro2 = st.columns([2, 1], gap="large")

with col_intro1:
    st.markdown("""
    <div class="info-card">
        <h3 class="info-title">Ce que je génère ici</h3>
        <p class="info-text">
            Le modèle V2 intègre des <strong>blocs CBAM</strong> (Convolutional Block Attention Module) 
            qui permettent au réseau de se concentrer sur les zones importantes de l'image. 
            Cette attention combinée (channel + spatial) améliore significativement la cohérence globale des visages générés.
        </p>
        <div class="info-specs">
            <div class="spec-item">
                <span class="spec-label">Architecture</span>
                <span class="spec-value">V1 + CBAM après chaque Conv2DTranspose</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Paramètres</span>
                <span class="spec-value">~2.8M (+12% vs V1)</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Vitesse</span>
                <span class="spec-value">~1.5s / 16 images</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_intro2:
    st.markdown("""
    <div class="tech-stack-card">
        <h4 class="tech-title">Mécanisme CBAM</h4>
        <p class="tech-description">
            Le bloc CBAM applique une double attention:
        </p>
        <div class="tech-tags">
            <span class="tech-tag tech-tag-highlight">Channel Attention</span>
            <span class="tech-tag tech-tag-highlight">Spatial Attention</span>
        </div>
        <p class="tech-description" style="margin-top: 1rem; font-size: 0.9rem;">
            Ratio: 8 | Kernel: 7×7
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Formulaire de génération
st.markdown('<div class="generation-section">', unsafe_allow_html=True)
st.markdown("## Configuration de génération")

simple_mode = st.toggle("Mode automatique", value=True, help="Détection automatique de z_dim")

with st.form("v2_form"):
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        weights = st.text_input(
            "Chemin du checkpoint",
            value="runs/v2/ckpt/generator_final.h5",
            help="Chemin vers le fichier .h5 du générateur V2"
        )
        n = st.slider("Nombre d'images", 4, 64, 16, step=4)
    
    with col_form2:
        seed = st.number_input("Seed aléatoire", value=42, step=1,
                              help="Pour reproductibilité des résultats")
        
        if not simple_mode:
            z_dim_manual = st.number_input("Dimension du bruit (z_dim)",
                                          value=100, min_value=8, max_value=1024, step=1)
        else:
            z_dim_manual = None
            st.info("z_dim détecté automatiquement depuis le modèle")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        submitted = st.form_submit_button("Générer les images", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Génération
if submitted:
    if not weights:
        st.error("Veuillez fournir un chemin de checkpoint valide")
    else:
        with st.spinner("Chargement du modèle V2 et génération..."):
            try:
                G = load_gan_generator(weights, attention=True)
                z_dim = get_generator_z_dim(G) if simple_mode else int(z_dim_manual)
                imgs = sample_gan(G, n=n, noise_dim=z_dim, seed=int(seed))
                grid = make_grid(list(imgs), ncols=4)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## Résultats de génération")
                
                col_result1, col_result2 = st.columns([3, 1])
                
                with col_result1:
                    st.image(grid, use_column_width=True)
                
                with col_result2:
                    st.markdown("""
                    <div class="result-info-card">
                        <h4>Informations</h4>
                        <div class="result-stat">
                            <span class="result-label">Images générées</span>
                            <span class="result-value">{}</span>
                        </div>
                        <div class="result-stat">
                            <span class="result-label">Dimension z</span>
                            <span class="result-value">{}</span>
                        </div>
                        <div class="result-stat">
                            <span class="result-label">Seed</span>
                            <span class="result-value">{}</span>
                        </div>
                        <div class="result-stat">
                            <span class="result-label">Attention</span>
                            <span class="result-value">CBAM</span>
                        </div>
                    </div>
                    """.format(n, z_dim, seed), unsafe_allow_html=True)
                    
                    from io import BytesIO
                    buf = BytesIO()
                    grid.save(buf, format="PNG")
                    st.download_button(
                        "Télécharger (PNG)",
                        data=buf.getvalue(),
                        file_name="v2_generation.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                st.success("Génération terminée avec succès!")
                
                # Comparaison avec V1
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Différences avec V1"):
                    st.markdown("""
                    **Améliorations apportées par CBAM:**
                    - Détails plus nets (yeux, bouche, contours)
                    - Meilleure symétrie faciale
                    - Cohérence globale améliorée
                    - FID Score: 45.8 (vs 52.3 pour V1)
                    """)
                
            except Exception as e:
                st.error(f"Erreur lors de la génération: {str(e)}")
                st.exception(e)