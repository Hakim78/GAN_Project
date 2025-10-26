import streamlit as st
from pathlib import Path
from fe_utils.loader import load_gan_generator, sample_gan, load_diffusion_model, sample_diffusion, get_generator_z_dim
from fe_utils.preview import make_grid
from fe_utils.compare import diversity_std
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import time

st.set_page_config(page_title="Comparaison", page_icon="⚖️", layout="wide")

# Je charge le CSS
css_file = Path(__file__).parents[1] / "ui" / "styles.css"
with open(css_file, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="compare-hero">
    <h1 class="compare-title">Comparaison des Architectures</h1>
    <p class="compare-subtitle">Analyse objective et visuelle de V1, V2 et V3</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    n = st.slider("Images par modèle", 4, 32, 16, step=4)
    seed = st.number_input("Seed", value=123, step=1)
    
    st.markdown("---")
    st.markdown("### Checkpoints")
    
    v1_ckpt = st.text_input("V1", value="runs/v1/ckpt/generator_final.h5")
    v2_ckpt = st.text_input("V2", value="runs/v2/ckpt/generator_final.h5")
    v3_ckpt = st.text_input("V3 (optionnel)", value="")
    
    st.markdown("---")
    st.markdown("### Options")
    
    show_stats = st.checkbox("Statistiques détaillées", value=True)
    show_histograms = st.checkbox("Histogrammes", value=True)

# Bouton de lancement
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_btn = st.button("Lancer la comparaison", use_container_width=True, type="primary")

if start_btn:
    results = {}
    timings = {}
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        # V1
        status.info("Génération V1...")
        progress_bar.progress(10)
        
        start = time.time()
        G1 = load_gan_generator(v1_ckpt, attention=False)
        z1 = get_generator_z_dim(G1)
        imgs1 = sample_gan(G1, n=n, noise_dim=z1, seed=int(seed))
        timings['v1'] = time.time() - start
        results['v1'] = {'images': imgs1, 'z_dim': z1, 'diversity': diversity_std(imgs1)}
        
        progress_bar.progress(33)
        
        # V2
        status.info("Génération V2...")
        start = time.time()
        G2 = load_gan_generator(v2_ckpt, attention=True)
        z2 = get_generator_z_dim(G2)
        imgs2 = sample_gan(G2, n=n, noise_dim=z2, seed=int(seed))
        timings['v2'] = time.time() - start
        results['v2'] = {'images': imgs2, 'z_dim': z2, 'diversity': diversity_std(imgs2)}
        
        progress_bar.progress(66)
        
        # V3 (optionnel)
        if v3_ckpt.strip():
            try:
                status.info("Génération V3 (Diffusion)...")
                start = time.time()
                D3 = load_diffusion_model(v3_ckpt)
                imgs3 = sample_diffusion(D3, n=n, steps=50, seed=int(seed))
                timings['v3'] = time.time() - start
                results['v3'] = {'images': imgs3, 'z_dim': 'N/A', 'diversity': diversity_std(imgs3)}
            except Exception as e:
                st.warning(f"V3 non disponible: {e}")
        
        progress_bar.progress(100)
        status.success("Génération terminée!")
        time.sleep(0.5)
        progress_bar.empty()
        status.empty()
        
        # Affichage des résultats
        st.markdown("---")
        st.markdown("## Résultats visuels")
        
        cols = st.columns(len(results))
        
        for idx, (version, data) in enumerate(results.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="compare-card">
                    <h3 class="compare-card-title">{version.upper()}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                grid = make_grid(list(data['images']), ncols=4)
                st.image(grid, use_column_width=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Diversité", f"{data['diversity']:.3f}")
                with col_b:
                    st.metric("Temps", f"{timings[version]:.2f}s")
                
                buf = BytesIO()
                grid.save(buf, format="PNG")
                st.download_button(
                    "Télécharger",
                    data=buf.getvalue(),
                    file_name=f"compare_{version}.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        # Statistiques
        if show_stats:
            st.markdown("---")
            st.markdown("## Tableau comparatif")
            
            import pandas as pd
            stats_data = []
            for version, data in results.items():
                imgs = data['images']
                stats_data.append({
                    'Modèle': version.upper(),
                    'z_dim': data['z_dim'],
                    'Images': len(imgs),
                    'Mean': f"{np.mean(imgs):.3f}",
                    'STD': f"{np.std(imgs):.3f}",
                    'Diversité': f"{data['diversity']:.3f}",
                    'Temps': f"{timings[version]:.2f}s"
                })
            
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Histogrammes
        if show_histograms:
            st.markdown("---")
            st.markdown("## Distribution des pixels")
            
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
            if len(results) == 1:
                axes = [axes]
            
            colors = ['#8A2BE2', '#00E5FF', '#FF6B6B']
            
            for idx, (version, data) in enumerate(results.items()):
                imgs_flat = data['images'].flatten()
                axes[idx].hist(imgs_flat, bins=50, alpha=0.7, 
                              color=colors[idx], edgecolor='black')
                axes[idx].set_title(f"{version.upper()}", fontweight='bold')
                axes[idx].set_xlabel('Valeur pixel')
                axes[idx].set_ylabel('Fréquence')
                axes[idx].grid(axis='y', alpha=0.3)
                axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Analyse
        st.markdown("---")
        st.markdown("## Analyse comparative")
        
        best_div = max(results.items(), key=lambda x: x[1]['diversity'])
        fastest = min(timings.items(), key=lambda x: x[1])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="analysis-card">
                <h4>Meilleure diversité</h4>
                <p class="analysis-value">{best_div[0].upper()}</p>
                <p class="analysis-detail">STD: {best_div[1]['diversity']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="analysis-card">
                <h4>Plus rapide</h4>
                <p class="analysis-value">{fastest[0].upper()}</p>
                <p class="analysis-detail">{fastest[1]:.2f} secondes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_quality = 'v3' if 'v3' in results else 'v2'
            st.markdown(f"""
            <div class="analysis-card">
                <h4>Meilleure qualité</h4>
                <p class="analysis-value">{best_quality.upper()}</p>
                <p class="analysis-detail">Basé sur FID scores</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommandations
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
        **Recommandations:**
        - **V1:** Prototypage rapide, tests initiaux
        - **V2:** Meilleur compromis qualité/vitesse
        - **V3:** Qualité maximale avec conditionnement
        """)
        
    except Exception as e:
        progress_bar.empty()
        status.empty()
        st.error(f"Erreur: {e}")
        st.exception(e)

# Section aide
with st.expander("Comment interpréter les résultats"):
    st.markdown("""
    ### Métriques
    
    **Diversité (STD):**
    - Plus élevé = plus de variabilité
    - Trop bas = mode collapse
    - Idéal: 0.1 - 0.3
    
    **Temps:**
    - GANs (V1/V2): ~1-2s
    - Diffusion (V3): ~20-30s
    
    **Distribution:**
    - Centrée sur 0 = bon
    - Uniforme = mauvais (bruit)
    """)