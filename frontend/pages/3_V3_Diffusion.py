import streamlit as st
from pathlib import Path
from fe_utils.loader import load_diffusion_model, sample_diffusion
from fe_utils.preview import make_grid

st.set_page_config(page_title="V3 – Diffusion", page_icon="🌫️", layout="wide")
with open(Path(__file__).parents[1] / "ui" / "styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="gradient-border"><h1>🌫️ V3 – Diffusion</h1></div>', unsafe_allow_html=True)

colA, colB = st.columns([2,1], gap="large")
with colA:
    weights = st.text_input("Chemin checkpoint (obligatoire)", value="", help="ex: runs/v3/ckpt/diffusion.h5")
    n = st.slider("Nombre d'images", 4, 64, 12, step=4)
    steps = st.slider("Sampling steps", 10, 500, 50, step=10)
    seed = st.number_input("Seed", value=42, step=1)
    gen_btn = st.button("🎨 Générer", type="primary", use_container_width=True)

with colB:
    st.markdown("""
    <div class="glass">
      <h3>Notes</h3>
      <ul>
        <li>Utilise ton vrai sampler (sample_fn) exposé par models.diffusion.</li>
        <li>Aucune simulation: échantillonnage réel uniquement.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

if gen_btn:
    if not weights:
        st.error("⚠️ Fournis un chemin de checkpoint valide.")
    else:
        try:
            D = load_diffusion_model(weights)
            imgs = sample_diffusion(D, n=n, steps=int(steps), seed=int(seed))
            grid = make_grid(list(imgs), ncols=4)
            st.image(grid, caption=f"Samples V3 (réel) – {steps} steps", use_column_width=True)
        except Exception as e:
            st.error(f"Erreur: {e}")
