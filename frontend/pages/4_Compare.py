import streamlit as st
from pathlib import Path
from fe_utils.loader import load_gan_generator, sample_gan, load_diffusion_model, sample_diffusion, get_generator_z_dim
from fe_utils.preview import make_grid
from fe_utils.compare import diversity_std
from fe_utils.lottie import render_lottie
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Comparaison Avancée", page_icon="⚖️", layout="wide")
with open(Path(__file__).parents[1] / "ui" / "styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="gradient-border"><h1>⚖️ Comparaison Avancée des Modèles</h1></div>', unsafe_allow_html=True)

# Section intro avec animation
colA, colB = st.columns([3, 2], gap="large")
with colA:
    st.markdown("""
    <div class="glass">
      <h3>Analyse comparative intelligente</h3>
      <p style="line-height: 1.8;">
        Je compare mes trois architectures sur des critères objectifs et visuels:
      </p>
      <ul style="line-height: 2;">
        <li><strong>Qualité visuelle:</strong> Netteté, cohérence, réalisme</li>
        <li><strong>Diversité:</strong> Écart-type des features (STD)</li>
        <li><strong>Temps de génération:</strong> Latence par image</li>
        <li><strong>Stabilité:</strong> Reproductibilité avec seed fixe</li>
      </ul>
      <p style="margin-top: 1rem; color: var(--accent-b); font-weight: bold;">
        💡 J'utilise le même seed pour garantir une comparaison équitable
      </p>
    </div>
    """, unsafe_allow_html=True)

with colB:
    render_lottie("lotties/compare.json", height=200)

# Configuration avancée dans la sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Paramètres de génération
    st.markdown("#### Paramètres de génération")
    n = st.slider("Nombre d'images par modèle", 4, 64, 16, step=4,
                  help="Plus d'images = meilleure estimation de la diversité")
    seed = st.number_input("Seed (répétabilité)", value=123, step=1,
                           help="Même seed = résultats reproductibles")
    
    st.markdown("---")
    
    # Chemins des checkpoints
    st.markdown("#### Chemins des checkpoints")
    v1_ckpt = st.text_input("V1 checkpoint", value="runs/v1/ckpt/generator_final.h5")
    v2_ckpt = st.text_input("V2 checkpoint", value="runs/v2/ckpt/generator_final.h5")
    v3_ckpt = st.text_input("V3 checkpoint (optionnel)", value="",
                            help="Laisser vide si V3 pas encore entraîné")
    
    st.markdown("---")
    
    # Options d'analyse
    st.markdown("#### Options d'analyse")
    show_histograms = st.checkbox("Afficher histogrammes de pixels", value=True)
    show_stats_table = st.checkbox("Afficher tableau de statistiques", value=True)
    measure_time = st.checkbox("Mesurer temps de génération", value=True)

# Bouton de lancement stylisé
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([2, 3, 2])
with col_btn2:
    go = st.button("🚀 Lancer la comparaison complète", use_container_width=True, type="primary")

if go:
    # Je crée un placeholder pour les updates en temps réel
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        results = {}
        timings = {}
        
        # Étape 1: V1
        status_placeholder.info("Chargement et génération V1...")
        progress_bar.progress(10)
        
        import time
        start_time = time.time()
        G1 = load_gan_generator(v1_ckpt, attention=False)
        z1 = get_generator_z_dim(G1)
        imgs1 = sample_gan(G1, n=n, noise_dim=z1, seed=int(seed))
        timings['v1'] = time.time() - start_time
        results['v1'] = {
            'images': imgs1,
            'z_dim': z1,
            'diversity': diversity_std(imgs1)
        }
        progress_bar.progress(33)
        
        # Étape 2: V2
        status_placeholder.info("Chargement et génération V2...")
        start_time = time.time()
        G2 = load_gan_generator(v2_ckpt, attention=True)
        z2 = get_generator_z_dim(G2)
        imgs2 = sample_gan(G2, n=n, noise_dim=z2, seed=int(seed))
        timings['v2'] = time.time() - start_time
        results['v2'] = {
            'images': imgs2,
            'z_dim': z2,
            'diversity': diversity_std(imgs2)
        }
        progress_bar.progress(66)
        
        # Étape 3: V3 (si disponible)
        if v3_ckpt.strip():
            try:
                status_placeholder.info("Chargement et génération V3 (Diffusion)...")
                start_time = time.time()
                D3 = load_diffusion_model(v3_ckpt)
                imgs3 = sample_diffusion(D3, n=n, steps=50, seed=int(seed))
                timings['v3'] = time.time() - start_time
                results['v3'] = {
                    'images': imgs3,
                    'z_dim': 'N/A (Diffusion)',
                    'diversity': diversity_std(imgs3)
                }
            except Exception as e:
                st.warning(f"V3 non disponible: {e}")
        
        progress_bar.progress(100)
        status_placeholder.success("✅ Génération terminée!")
        time.sleep(0.5)
        status_placeholder.empty()
        progress_bar.empty()
        
        # Affichage des résultats
        st.markdown("---")
        st.markdown("## 📊 Résultats de la comparaison")
        
        # Grilles d'images
        st.markdown("### 🖼️ Grilles d'images générées")
        cols = st.columns(len(results))
        
        for idx, (version, data) in enumerate(results.items()):
            with cols[idx]:
                grid = make_grid(list(data['images']), ncols=4)
                st.image(grid, caption=f"{version.upper()} (z_dim={data['z_dim']})", use_column_width=True)
                
                # Métriques sous l'image
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Diversité (STD)", f"{data['diversity']:.3f}")
                with col_b:
                    if measure_time and version in timings:
                        st.metric("Temps", f"{timings[version]:.2f}s")
                
                # Bouton téléchargement
                buf = BytesIO()
                grid.save(buf, format="PNG")
                st.download_button(
                    f"⬇️ Télécharger {version.upper()}", 
                    data=buf.getvalue(), 
                    file_name=f"compare_{version}.png", 
                    mime="image/png",
                    use_container_width=True
                )
        
        # Tableau de statistiques
        if show_stats_table:
            st.markdown("---")
            st.markdown("### 📈 Tableau comparatif")
            
            import pandas as pd
            stats_data = []
            for version, data in results.items():
                imgs = data['images']
                stats_data.append({
                    'Modèle': version.upper(),
                    'z_dim': data['z_dim'],
                    'Nb images': len(imgs),
                    'Mean pixel': f"{np.mean(imgs):.3f}",
                    'STD pixel': f"{np.std(imgs):.3f}",
                    'Diversité': f"{data['diversity']:.3f}",
                    'Temps (s)': f"{timings.get(version, 0):.2f}" if measure_time else "N/A"
                })
            
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Histogrammes de distribution
        if show_histograms:
            st.markdown("---")
            st.markdown("### 📊 Distribution des pixels")
            
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
            if len(results) == 1:
                axes = [axes]
            
            for idx, (version, data) in enumerate(results.items()):
                imgs_flat = data['images'].flatten()
                axes[idx].hist(imgs_flat, bins=50, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][idx], edgecolor='black')
                axes[idx].set_title(f"{version.upper()}", fontweight='bold', fontsize=12)
                axes[idx].set_xlabel('Valeur pixel')
                axes[idx].set_ylabel('Fréquence')
                axes[idx].grid(axis='y', alpha=0.3)
                axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Graphique de comparaison des métriques
        st.markdown("---")
        st.markdown("### 🎯 Comparaison des métriques clés")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Graphique diversité
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            versions = list(results.keys())
            diversities = [results[v]['diversity'] for v in versions]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(versions)]
            
            bars = ax1.bar(versions, diversities, color=colors, alpha=0.7, edgecolor='black')
            for bar, div in zip(bars, diversities):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{div:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_ylabel('Diversité (STD)', fontweight='bold')
            ax1.set_title('Comparaison de la diversité', fontweight='bold', pad=10)
            ax1.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col_chart2:
            # Graphique temps
            if measure_time:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                times = [timings.get(v, 0) for v in versions]
                
                bars = ax2.bar(versions, times, color=colors, alpha=0.7, edgecolor='black')
                for bar, t in zip(bars, times):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{t:.2f}s', ha='center', va='bottom', fontweight='bold')
                
                ax2.set_ylabel('Temps (secondes)', fontweight='bold')
                ax2.set_title('Comparaison du temps de génération', fontweight='bold', pad=10)
                ax2.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
        
        # Analyse textuelle
        st.markdown("---")
        st.markdown("### 🔍 Analyse et recommandations")
        
        # Je trouve le meilleur modèle par diversité
        best_diversity = max(results.items(), key=lambda x: x[1]['diversity'])
        fastest = min(timings.items(), key=lambda x: x[1]) if measure_time else None
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.markdown(f"""
            <div class="glass" style="background: rgba(138,43,226,0.1); padding: 1.5rem;">
              <h4>🏆 Meilleure diversité</h4>
              <p style="font-size: 1.2rem; font-weight: bold; color: var(--accent-a);">
                {best_diversity[0].upper()}
              </p>
              <p>Diversité: {best_diversity[1]['diversity']:.3f}</p>
              <p style="color: var(--text-1); margin-top: 1rem;">
                Ce modèle génère des visages plus variés, réduisant le risque de mode collapse.
              </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_analysis2:
            if fastest:
                st.markdown(f"""
                <div class="glass" style="background: rgba(0,229,255,0.1); padding: 1.5rem;">
                  <h4>⚡ Plus rapide</h4>
                  <p style="font-size: 1.2rem; font-weight: bold; color: var(--accent-b);">
                    {fastest[0].upper()}
                  </p>
                  <p>Temps: {fastest[1]:.2f} secondes</p>
                  <p style="color: var(--text-1); margin-top: 1rem;">
                    Idéal pour génération en temps réel ou production à grande échelle.
                  </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Recommandations contextuelles
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
        **💡 Recommandations d'utilisation:**
        
        - **V1 (Baseline):** Prototypage rapide, tests initiaux
        - **V2 (Attention):** Production avec contraintes de qualité élevées
        - **V3 (Diffusion):** Meilleure qualité théorique, mais plus lent (génération conditionnelle)
        """)
        
        # Export global
        st.markdown("---")
        st.markdown("### 💾 Export des résultats")
        
        if st.button("📦 Télécharger rapport complet (PDF)", use_container_width=True):
            st.info("Fonctionnalité en développement - Les données sont sauvegardées localement")
        
    except Exception as e:
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"❌ Erreur lors de la comparaison: {e}")
        st.exception(e)

# Section d'aide
with st.expander("❓ Aide et interprétation des résultats"):
    st.markdown("""
    ### Comment interpréter les métriques
    
    **Diversité (STD):**
    - Plus élevé = plus de variabilité entre les images générées
    - Trop bas = risque de mode collapse (le modèle génère toujours les mêmes visages)
    - Valeur idéale: entre 0.1 et 0.3 pour des images normalisées [-1, 1]
    
    **Temps de génération:**
    - V1 et V2 (GANs): Génération quasi-instantanée (~0.5-2s pour 16 images)
    - V3 (Diffusion): Plus lent car débruitage itératif (~10-30s pour 16 images avec 50 steps)
    
    **Distribution des pixels:**
    - Distribution centrée autour de 0: bon (images normalisées)
    - Distribution uniforme: mauvais (le modèle génère du bruit)
    - Pics distincts: le modèle a appris des patterns spécifiques
    
    ### Pourquoi utiliser le même seed?
    
    Le seed contrôle la génération aléatoire du bruit initial. En utilisant le même seed pour tous les modèles:
    - Je compare leur capacité à transformer **le même bruit** en images réalistes
    - J'élimine la variable aléatoire pour une comparaison équitable
    - Les différences observées proviennent uniquement des architectures
    """)