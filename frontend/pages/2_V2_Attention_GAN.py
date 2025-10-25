import streamlit as st
from pathlib import Path
from fe_utils.loader import load_gan_generator, sample_gan, get_generator_z_dim
from fe_utils.preview import make_grid
from fe_utils.lottie import render_lottie

st.set_page_config(page_title="V2 – Attention GAN", page_icon=None, layout="wide")
with open(Path(__file__).parents[1] / "ui" / "styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="gradient-border"><h1>V2 – GAN avec Attention</h1></div>', unsafe_allow_html=True)

# Intro grand public
colA, colB = st.columns([2,1], gap="large")
with colA:
    st.markdown("""
    <div class="glass">
      <h3>Ce que je génère ici</h3>
      <p>Le modèle V2 ajoute un mécanisme d’<b>Attention</b> qui aide à mieux organiser l’image 
      (par exemple, la cohérence globale d’un visage). Le point de départ reste le <b>bruit</b>, 
      mais l’Attention permet souvent des détails plus nets.</p>
    </div>
    """, unsafe_allow_html=True)
with colB:
    render_lottie("lotties/v2.json", height=160)

# Paramètres – mode simple / expert
st.markdown("### Paramètres")
simple_mode = st.toggle("Mode simple (recommandé)", value=True, help="Je laisse l’app détecter la bonne dimension de bruit (z_dim).")

with st.form("v2_form"):
    weights = st.text_input(
        "Chemin checkpoint (obligatoire)",
        value="runs/v2/ckpt/generator_final.h5",
        help="ex: runs/v2/ckpt/generator_final.h5"
    )
    n = st.slider("Nombre d'images à générer", 4, 64, 12, step=4)
    seed = st.number_input("Seed (répétabilité du résultat)", value=42, step=1)

    if simple_mode:
        st.caption("Le z_dim est détecté automatiquement depuis le modèle.")
        z_dim_manual = None
    else:
        z_dim_manual = st.number_input("z_dim (dimension du bruit)", value=100, min_value=8, max_value=1024, step=1,
                                       help="Valeur utilisée lors de l’entraînement.")

    submitted = st.form_submit_button("Générer", use_container_width=True)

if submitted:
    if not weights:
        st.error("Fournis un chemin de checkpoint valide.")
    else:
        try:
            G = load_gan_generator(weights, attention=True)
            z_dim = get_generator_z_dim(G) if simple_mode else int(z_dim_manual)
            imgs = sample_gan(G, n=n, noise_dim=z_dim, seed=int(seed))
            grid = make_grid(list(imgs), ncols=4)
            st.image(grid, caption=f"V2 – {n} images (z_dim={z_dim})", use_column_width=True)

            # Bouton de téléchargement
            from io import BytesIO
            buf = BytesIO()
            grid.save(buf, format="PNG")
            st.download_button("Télécharger l’image (PNG)", data=buf.getvalue(), file_name="v2_grid.png", mime="image/png")
        except Exception as e:
            st.error(f"Erreur: {e}")
