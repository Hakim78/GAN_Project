import streamlit as st
from pathlib import Path
from fe_utils.lottie import render_lottie

st.set_page_config(page_title="Studio Génératif – Demo", page_icon="🎬", layout="wide")

# CSS
with open(Path(__file__).parent / "ui" / "styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="gradient-border"><h1>Studio Génératif – Démonstration</h1></div>', unsafe_allow_html=True)

# Bandeau de valeur (à quoi ça sert ?)
col1, col2 = st.columns([1,1])
with col1:
    st.markdown("""
    <div class="glass">
      <h3>À quoi sert ce site ?</h3>
      <p>Je présente une démonstration interactive de mes modèles génératifs entraînés en interne :</p>
      <ul>
        <li><b>V1</b> : GAN baseline</li>
        <li><b>V2</b> : GAN avec Attention</li>
        <li><b>V3</b> : Modèle de Diffusion (optionnel si checkpoint disponible)</li>
      </ul>
      <p>Objectif : montrer comment générer rapidement des visuels originaux (portraits stylisés, assets créatifs) 
      à destination de maquettes, miniatures, posts, ou contenus courts (TikTok/YouTube Shorts).</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    render_lottie("lotties/home.json", height=220)

# Tutoriel simple étape par étape
st.markdown("""
<div class="glass">
  <h3>Comment l’utiliser en 2 minutes</h3>
  <ol>
    <li>Ouvrir la page <b>V1</b> ou <b>V2</b> dans le menu de gauche.</li>
    <li>Laisser le chemin de checkpoint par défaut (ou coller un autre chemin si besoin).</li>
    <li>Cliquer sur <b>Générer</b> : je récupère des images <i>réelles</i> de mes modèles.</li>
    <li>Aller sur <b>Compare</b> pour afficher V1 vs V2 (et V3 si dispo) sur la même configuration.</li>
  </ol>
  <p><i>Remarque :</i> tout est local à mes checkpoints. Aucune donnée factice, aucune simulation.</p>
</div>
""", unsafe_allow_html=True)

# Blocs explicatifs (vulgarisation)
st.markdown("### Comprendre sans jargon")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="glass">
      <h4>V1 – GAN baseline</h4>
      <p>Un GAN apprend à créer des images en opposant deux réseaux : 
      un <b>Générateur</b> (qui propose des images) et un <b>Discriminateur</b> (qui juge si elles semblent réelles).</p>
    </div>
    """, unsafe_allow_html=True)
    render_lottie("lotties/v1.json", height=180)
with c2:
    st.markdown("""
    <div class="glass">
      <h4>V2 – GAN avec Attention</h4>
      <p>L’<b>Attention</b> aide le modèle à mieux capter les relations globales (ex. cohérence du visage complet), 
      ce qui améliore la netteté et la stabilité de certains détails.</p>
    </div>
    """, unsafe_allow_html=True)
    render_lottie("lotties/v2.json", height=180)
with c3:
    st.markdown("""
    <div class="glass">
      <h4>Compare</h4>
      <p>Je génère le même nombre d’images pour chaque version et j’affiche un petit indicateur de diversité 
      pour illustrer la différence de comportement.</p>
    </div>
    """, unsafe_allow_html=True)
    render_lottie("lotties/compare.json", height=180)

# Appel à l’action
st.markdown("""
<div class="glass">
  <h3>Commencer</h3>
  <ul>
    <li>V1 : <b>GAN Baseline</b> – simple et rapide.</li>
    <li>V2 : <b>GAN avec Attention</b> – meilleure cohérence globale.</li>
    <li>Compare : j’affiche les deux côte à côte (et V3 si j’ai un checkpoint de diffusion).</li>
  </ul>
  <p>Je vais dans le menu à gauche pour choisir ma page.</p>
</div>
""", unsafe_allow_html=True)
