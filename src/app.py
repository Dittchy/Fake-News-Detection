import streamlit as st
import os
import torch
import shap
import streamlit.components.v1 as components
import numpy as np
# Import logic from existing modules
# Note: These imports work when running from root directory: streamlit run src/app.py
import sys
from pathlib import Path
# Add the project root to the python path so imports from 'src' work correctly
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import FakeNewsEnsemble

# --- 1. SETUP PAGE CONFIGURATION ---
# This sets the title and icon you see in the browser tab.
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

# --- 2. CUSTOM STYLING (THEME: SPY/DARK/DETECTIVE) ---
import base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_img = get_base64("background.png")
except:
    bg_img = ""

st.markdown(f"""
<style>
    /* Animation Keyframes */
    @keyframes rain {{
        0% {{ background-position: 0% 0%; }}
        100% {{ background-position: 0% 100%; }}
    }}
    
    @keyframes breathe {{
        0% {{ background-size: 100% 100%; }}
        50% {{ background-size: 105% 105%; }}
        100% {{ background-size: 100% 100%; }}
    }}

    /* Background Image with Filters */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        /* Sharpening effect via Contrast */
        filter: contrast(1.2) brightness(0.9); 
    }}
    
    /* Rain & Overlay Layer */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        /* Dark Gradient + Scanlines */
        background: 
            linear-gradient(rgba(10, 10, 10, 0.40), rgba(10, 10, 10, 0.50)),
            repeating-linear-gradient(0deg, transparent, transparent 2px, #000 3px);
        background-size: cover, 100% 4px;
        z-index: -1;
    }}
    
    /* Digital Rain / Static Animation Layer */
    .stApp::after {{
        content: "";
        position: absolute;
        top: 0; 
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("https://www.transparenttextures.com/patterns/stardust.png"); /* Noise texture */
        opacity: 0.1;
        animation: rain 10s linear infinite;
        pointer-events: none;
        z-index: -1;
    }}

    /* Global Text Color - Sharper */
    .stApp, p, h1, h2, h3, label, .stMarkdown {{
        color: #e0e0e0 !important;
        font-family: 'Courier New', Courier, monospace;
        text-shadow: 1px 1px 2px black;
    }}
    
    /* Text Area (Dossier Input) */
    .stTextArea textarea {{
        background-color: rgba(31, 35, 41, 0.8) !important; /* Semi-transparent */
        color: #00ff41 !important;
        border: 1px solid #555;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: rgba(211, 47, 47, 0.9) !important;
        color: white !important;
        border: 1px solid #ff2b2b;
        backdrop-filter: blur(5px);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid #333;
    }}
    
    /* Metrics */
    div[data-testid="metric-container"] {{
        background-color: rgba(38, 39, 48, 0.8);
        border: 1px solid #555;
        backdrop-filter: blur(5px);
    }}
</style>
""", unsafe_allow_html=True)

# Main Title with Icon
st.title("🕵️ PROJECT: TRUTH SEEKER")
st.markdown("<h3 style='color: #888;'>// CLASSIFIED INTELLIGENCE SYSTEM //</h3>", unsafe_allow_html=True)
st.divider()

# --- 3. LOAD MODELS ---
# We use @st.cache_resource so we only load the heavy AI models ONCE.
# Otherwise, the app would reload them every time you click a button (very slow!).
@st.cache_resource
def load_ensemble():
    return FakeNewsEnsemble()

try:
    ensemble = load_ensemble()
    st.sidebar.success("✅ Models Loaded Successfully!") # Show success message in sidebar
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 4. USER INTERFACE ---
# Create a big text box for the user to paste news.
st.markdown("#### 📄 PASTE SUSPECT DOSSIER BELOW:")
text_input = st.text_area("", 
                          height=200, 
                          placeholder="Awaiting encrypted transmission...")

# Layout: Create 3 columns to center the button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # A big primary button to trigger detection
    analyze_btn = st.button("INITIATE SURVEILLANCE SCAN 🦅", use_container_width=True, type="primary")

if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ NO DATA DETECTED. ABORTING SCAN.")
    else:
        with st.spinner("Decrypting... Analyzing Patterns... Accessing Neural Net..."):
            # 1. Prediction
            try:
                score = ensemble.predict(text_input)
                
                # Display Result
                st.divider()
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.subheader("Verdict")
                    if score > 0.5:
                        st.markdown("<h2 style='color: #ff2b2b; border: 2px solid red; padding: 10px; text-align: center;'>🚫 DECEPTION DETECTED</h2>", unsafe_allow_html=True)
                        st.metric("CONFIDENCE LEVEL", f"{score*100:.2f}%")
                    else:
                        st.markdown("<h2 style='color: #00ff41; border: 2px solid green; padding: 10px; text-align: center;'>✅ VERIFIED TRUTH</h2>", unsafe_allow_html=True)
                        st.metric("CONFIDENCE LEVEL", f"{(1-score)*100:.2f}%")
                
                with res_col2:
                    st.subheader("Forensic Breakdown")
                    st.write("Results from Neural Agents:")
                    st.progress(score, text=f"Deception Probability: {score:.2f}")
                    st.write("- **Agent RoBERTa (Transformer)**: 80% Weight")
                    st.write("- **Agent SVM (Baseline)**: 10% Weight")
                    st.write("- **Agent GNN (Network)**: 10% Weight")

                # 2. Explainability
                st.divider()
                st.subheader("🔍 Explainability (SHAP)")
                st.write("Highlighting words that contributed most to the decision (Red = Fake, Blue = Real).")
                
                with st.spinner("Generating explanation..."):
                    # We replicate SHAP logic here to render in Streamlit
                    tokenizer = ensemble.bert_tokenizer
                    model = ensemble.bert_model
                    
                    if model and tokenizer:
                        # Wrapper
                        def predictor(texts):
                            if isinstance(texts, str): texts = [texts]
                            elif isinstance(texts, np.ndarray): texts = texts.tolist()
                            texts = [str(t) for t in texts]

                            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
                            inputs = {k: v.to(ensemble.device) for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
                            return probs

                        masker = shap.maskers.Text(tokenizer)
                        explainer = shap.Explainer(predictor, masker, output_names=["Fake", "Real"])
                        shap_values = explainer([text_input])
                        
                        # Render HTML
                        # Force white text/background fix via wrapping div
                        # Visualize attribution for "Fake" class (Index 0)
                        html = shap.plots.text(shap_values[..., 0], display=False)
                        
                        st.components.v1.html(f"""
                        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid #333; color: #000000; font-family: 'Courier New'; opacity: 0.95;">
                            <h3 style="color: #000000; margin-bottom: 10px; border-bottom: 2px solid #333; text-transform: uppercase;">📂 EVIDENCE BOARD (DECRYPTED)</h3>
                            <p style="font-size: 14px; color: #333; font-weight: bold;">RED = FAKE INDICATOR | BLUE = REAL INDICATOR</p>
                            
                            <!-- No inversion, just clean white paper look for maximum readability -->
                            <div style="margin-top: 15px;"> 
                             {html} 
                            </div>
                        </div>
                        """, height=500, scrolling=True)
                    else:
                        st.warning("RoBERTa model not loaded, cannot generate SHAP explanation.")
            
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

st.sidebar.info("This project uses a Hybrid Ensemble of **SVM**, **RoBERTa**, and **GNN** to detect fake news with high accuracy.")
st.sidebar.markdown("---")
st.sidebar.write("Created for Fake News Detection Project")
