import streamlit as st
from pathlib import Path
from fe_utils.loader import load_gan_generator, sample_gan, load_diffusion_model, sample_diffusion, get_generator_z_dim
from fe_utils.preview import make_grid
from fe_utils.compare import diversity_std
from fe_utils.lottie import render_lottie

st.set_page_config(page_title="Comparaison", page_icon=None, layout="wide")
with open(Path(__file__).parents[1] / "ui" / "styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="gradient-border"><h1>Comparer les versions</h1></div>', unsafe_allow_html=True)

colA, colB = st.columns([2,1], gap="large")
with colA:
    st.markdown("""
    <div class="glass">
      <h3>Comment je compare</h3>
      <p>Je génère le même nombre d’images pour chaque modèle avec une configuration équivalente (même seed, 
      z_dim détecté automatiquement). J’affiche ensuite les grilles côte à côte et un indicateur simple de diversité.</p>
      <p>But : donner une lecture rapide des différences visuelles entre V1, V2 (et V3 si j’ai un checkpoint).</p>
    </div>
    """, unsafe_allow_html=True)
with colB:
    render_lottie("lotties/compare.json", height=160)

with st.sidebar:
    st.markdown("### Paramètres")
    n = st.slider("Nombre d'images par modèle", 8, 64, 16, step=8)
    seed = st.number_input("Seed (répétabilité du résultat)", value=123, step=1)
    st.markdown("---")
    v1_ckpt = st.text_input("V1 ckpt", value="runs/v1/ckpt/generator_final.h5")
    v2_ckpt = st.text_input("V2 ckpt", value="runs/v2/ckpt/generator_final.h5")
    v3_ckpt = st.text_input("V3 ckpt (optionnel)", value="")

go = st.button("Lancer la comparaison", use_container_width=True)

if go:
    try:
        cols = st.columns(3)
        # V1
        with cols[0]:
            G1 = load_gan_generator(v1_ckpt, attention=False)
            z1 = get_generator_z_dim(G1)
            imgs1 = sample_gan(G1, n=n, noise_dim=z1, seed=int(seed))
            grid1 = make_grid(list(imgs1), ncols=4)
            st.image(grid1, caption=f"V1 (z_dim={z1})", use_column_width=True)
            st.metric("Diversité (STD)", f"{diversity_std(imgs1):.3f}")
            from io import BytesIO
            b1 = BytesIO(); grid1.save(b1, format="PNG")
            st.download_button("Télécharger V1", data=b1.getvalue(), file_name="compare_v1.png", mime="image/png")

        # V2
        with cols[1]:
            G2 = load_gan_generator(v2_ckpt, attention=True)
            z2 = get_generator_z_dim(G2)
            imgs2 = sample_gan(G2, n=n, noise_dim=z2, seed=int(seed))
            grid2 = make_grid(list(imgs2), ncols=4)
            st.image(grid2, caption=f"V2 (z_dim={z2})", use_column_width=True)
            st.metric("Diversité (STD)", f"{diversity_std(imgs2):.3f}")
            from io import BytesIO
            b2 = BytesIO(); grid2.save(b2, format="PNG")
            st.download_button("Télécharger V2", data=b2.getvalue(), file_name="compare_v2.png", mime="image/png")

        # V3 (si fourni)
        with cols[2]:
            if v3_ckpt.strip():
                try:
                    D3 = load_diffusion_model(v3_ckpt)
                    imgs3 = sample_diffusion(D3, n=n, steps=50, seed=int(seed))
                    grid3 = make_grid(list(imgs3), ncols=4)
                    st.image(grid3, caption="V3 (Diffusion)", use_column_width=True)
                    st.metric("Diversité (STD)", f"{diversity_std(imgs3):.3f}")
                    from io import BytesIO
                    b3 = BytesIO(); grid3.save(b3, format="PNG")
                    st.download_button("Télécharger V3", data=b3.getvalue(), file_name="compare_v3.png", mime="image/png")
                except Exception as e:
                    st.error(f"Erreur V3: {e}")
            else:
                st.info("Aucun checkpoint V3 fourni.")
    except Exception as e:
        st.error(f"Erreur: {e}")
