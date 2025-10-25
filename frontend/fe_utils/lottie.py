from pathlib import Path
import json
import streamlit as st
import streamlit.components.v1 as components

def load_lottie_json(path: str):
    """Charge un JSON Lottie depuis un fichier local."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def render_lottie(path: str, height: int = 220, speed: float = 1.0, loop: bool = True, autoplay: bool = True):
    """
    Affiche un Lottie local en utilisant le web-component <lottie-player>.
    Nécessite une connexion Internet pour charger le script du player (CDN).
    """
    data = load_lottie_json(path)
    if data is None:
        st.info(f"Lottie introuvable ou invalide : {path}")
        return

    # On écrit le JSON dans la page et on le lit via JS
    json_id = f"lottie_json_{hash(path)}"
    st.markdown(
        f'<script id="{json_id}" type="application/json">{json.dumps(data)}</script>',
        unsafe_allow_html=True
    )

    loop_str = "true" if loop else "false"
    autoplay_str = "true" if autoplay else "false"

    html = f"""
    <div style="display:flex;justify-content:center;">
      <lottie-player
        id="player_{json_id}"
        style="height:{height}px;"
        speed="{speed}"
        loop="{loop_str}"
        autoplay="{autoplay_str}">
      </lottie-player>
    </div>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <script>
      (function() {{
        const script = document.getElementById("{json_id}");
        if (!script) return;
        try {{
          const data = JSON.parse(script.textContent);
          const player = document.getElementById("player_{json_id}");
          if (player) {{
            player.load(JSON.stringify(data));
          }}
        }} catch (e) {{
          console.error("Lottie load error", e);
        }}
      }})();
    </script>
    """
    components.html(html, height=height+20, scrolling=False)
