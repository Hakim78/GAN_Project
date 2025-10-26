import streamlit as st
from pathlib import Path
from fe_utils.loader import load_diffusion_model, sample_diffusion
from fe_utils.preview import make_grid
import base64

st.set_page_config(page_title="V3 – Diffusion", page_icon="🌟", layout="wide")

# Je charge le CSS
css_file = Path(__file__).parents[1] / "ui" / "styles.css"
with open(css_file, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Je charge l'image de background V3
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_v3 = get_image_base64(Path(__file__).parents[1] / "assets" / "img" / "v3_background_img.png")

# Hero avec background image
st.markdown(f"""
<div class="page-hero" style="background-image: url('data:image/png;base64,{bg_v3}');">
    <div class="page-hero-overlay"></div>
    <div class="page-hero-content">
        <div class="page-badge page-badge-accent">Version 3</div>
        <h1 class="page-title">Modèle de Diffusion (DDPM)</h1>
        <p class="page-subtitle">Génération conditionnelle par débruitage progressif</p>
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
            Le modèle V3 utilise un <strong>U-Net conditionnel</strong> pour générer des images par débruitage progressif (DDPM).
            Contrairement aux GANs, cette approche est plus stable et permet un contrôle précis des attributs générés
            (genre, sourire, lunettes, etc.) grâce au conditionnement sur les 40 attributs CelebA.
        </p>
        <div class="info-specs">
            <div class="spec-item">
                <span class="spec-label">Architecture</span>
                <span class="spec-value">U-Net + Time Embedding + Condition</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Paramètres</span>
                <span class="spec-value">~5.2M</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Steps</span>
                <span class="spec-value">1000 (entraînement) / 50-200 (génération)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_intro2:
    st.markdown("""
    <div class="tech-stack-card">
        <h4 class="tech-title">Processus DDPM</h4>
        <p class="tech-description">
            Débruitage itératif:
        </p>
        <div class="diffusion-process">
            <div class="process-step">Bruit pur (t=T)</div>
            <div class="process-arrow">↓</div>
            <div class="process-step">Débruitage progressif</div>
            <div class="process-arrow">↓</div>
            <div class="process-step">Image finale (t=0)</div>
        </div>
        <div class="tech-tags" style="margin-top: 1rem;">
            <span class="tech-tag tech-tag-highlight">Conditionnel</span>
            <span class="tech-tag">FID: 38.2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Formulaire de génération
st.markdown('<div class="generation-section">', unsafe_allow_html=True)
st.markdown("## Configuration de génération")

with st.form("v3_form"):
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        weights = st.text_input(
            "Chemin du checkpoint",
            value="runs/v3/ckpt/unet_final.h5",
            help="Chemin vers le fichier .h5 du U-Net"
        )
        n = st.slider("Nombre d'images", 4, 32, 12, step=4,
                     help="Génération plus lente que V1/V2")
    
    with col_form2:
        steps = st.slider("Steps de débruitage", 10, 200, 50, step=10,
                         help="Plus de steps = meilleure qualité mais plus lent")
        seed = st.number_input("Seed aléatoire", value=42, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Avertissement
    st.warning("""
    **Note importante:** La génération avec V3 prend plus de temps (~20-30s pour 12 images avec 50 steps).
    La qualité est supérieure mais le processus est plus lent que les GANs.
    """)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        submitted = st.form_submit_button("Générer les images", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Génération
if submitted:
    if not weights:
        st.error("Veuillez fournir un chemin de checkpoint valide")
    else:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            progress_text.text("Chargement du U-Net...")
            progress_bar.progress(10)
            
            D = load_diffusion_model(weights)
            
            progress_text.text(f"Génération en cours ({steps} steps de débruitage)...")
            progress_bar.progress(30)
            
            imgs = sample_diffusion(D, n=n, steps=int(steps), seed=int(seed))
            
            progress_bar.progress(100)
            progress_text.text("Génération terminée!")
            
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
                        <span class="result-label">Steps</span>
                        <span class="result-value">{}</span>
                    </div>
                    <div class="result-stat">
                        <span class="result-label">Seed</span>
                        <span class="result-value">{}</span>
                    </div>
                    <div class="result-stat">
                        <span class="result-label">Méthode</span>
                        <span class="result-value">DDPM</span>
                    </div>
                </div>
                """.format(n, steps, seed), unsafe_allow_html=True)
                
                from io import BytesIO
                buf = BytesIO()
                grid.save(buf, format="PNG")
                st.download_button(
                    "Télécharger (PNG)",
                    data=buf.getvalue(),
                    file_name="v3_diffusion.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            progress_text.empty()
            progress_bar.empty()
            st.success("Génération terminée avec succès!")
            
            # Avantages de la diffusion
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Pourquoi utiliser la diffusion?"):
                st.markdown("""
                **Avantages par rapport aux GANs:**
                - **Stabilité:** Pas de mode collapse, entraînement plus stable
                - **Qualité:** Meilleur FID score (38.2 vs 45.8 pour V2)
                - **Contrôle:** Conditionnement natif sur attributs
                - **Diversité:** Meilleure couverture de l'espace des données
                
                **Inconvénients:**
                - **Vitesse:** Génération plus lente (débruitage itératif)
                - **Complexité:** Plus de paramètres et de mémoire
                """)
            
        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            st.error(f"Erreur lors de la génération: {str(e)}")
            st.exception(e)