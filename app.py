import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Parkinson Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/parkinsons_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Remove top white spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Hide Streamlit header */
header {
    visibility: hidden;
}

/* Hide toolbar */
[data-testid="stToolbar"] {
    visibility: hidden;
}

/* Title */
h1 {
    color: #4FC3F7 !important;
    text-align: center;
    font-size: 50px !important;
}

/* Labels */
label {
    color: white !important;
    font-weight: 600 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Button */
.stButton>button {
    background-color: #4FC3F7;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #29B6F6;
    color: white;
}

/* Success box */
.success-box {
    background-color: #1B5E20;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

/* Error box */
.error-box {
    background-color: #B71C1C;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Parkinson Prediction")

st.sidebar.info(
    "This system predicts Parkinson Disease using "
    "Machine Learning and voice analysis."
)



# ---------------- TITLE ----------------
st.title("Parkinson Disease Prediction System")

st.write("### Load Sample Data")

# ---------------- SAMPLE BUTTONS ----------------

col_btn1, col_btn2 = st.columns(2)

with col_btn1:

    if st.button("✅ Load Healthy Sample"):

        st.session_state.fo = 214.289
        st.session_state.fhi = 260.277
        st.session_state.flo = 77.973
        st.session_state.jitter_percent = 0.00567
        st.session_state.jitter_abs = 0.00003
        st.session_state.rap = 0.00295
        st.session_state.ppq = 0.00317
        st.session_state.ddp = 0.00885
        st.session_state.shimmer = 0.01884
        st.session_state.shimmer_db = 0.19
        st.session_state.apq3 = 0.01026
        st.session_state.apq5 = 0.01161
        st.session_state.apq = 0.01373
        st.session_state.dda = 0.03078
        st.session_state.nhr = 0.04398
        st.session_state.hnr = 21.209
        st.session_state.rpde = 0.462803
        st.session_state.dfa = 0.664357
        st.session_state.spread1 = -5.724056
        st.session_state.spread2 = 0.190667
        st.session_state.d2 = 2.555477
        st.session_state.ppe = 0.148569

with col_btn2:

    if st.button("⚠ Load Parkinson Sample"):

        st.session_state.fo = 119.992
        st.session_state.fhi = 157.302
        st.session_state.flo = 74.997
        st.session_state.jitter_percent = 0.00784
        st.session_state.jitter_abs = 0.00007
        st.session_state.rap = 0.00370
        st.session_state.ppq = 0.00554
        st.session_state.ddp = 0.01109
        st.session_state.shimmer = 0.04374
        st.session_state.shimmer_db = 0.426
        st.session_state.apq3 = 0.02182
        st.session_state.apq5 = 0.03130
        st.session_state.apq = 0.02971
        st.session_state.dda = 0.06545
        st.session_state.nhr = 0.02211
        st.session_state.hnr = 21.033
        st.session_state.rpde = 0.414783
        st.session_state.dfa = 0.815285
        st.session_state.spread1 = -4.813031
        st.session_state.spread2 = 0.266482
        st.session_state.d2 = 2.301442
        st.session_state.ppe = 0.284654

# ---------------- INPUT FIELDS ----------------

st.write("### Enter Voice Measurement Values")

col1, col2 = st.columns(2)

with col1:

    fo = st.number_input(
        "MDVP:Fo(Hz)",
        value=st.session_state.get("fo", 0.0),
        format="%.6f"
    )

    fhi = st.number_input(
        "MDVP:Fhi(Hz)",
        value=st.session_state.get("fhi", 0.0),
        format="%.6f"
    )

    flo = st.number_input(
        "MDVP:Flo(Hz)",
        value=st.session_state.get("flo", 0.0),
        format="%.6f"
    )

    jitter_percent = st.number_input(
        "MDVP:Jitter(%)",
        value=st.session_state.get("jitter_percent", 0.0),
        format="%.6f"
    )

    jitter_abs = st.number_input(
        "MDVP:Jitter(Abs)",
        value=st.session_state.get("jitter_abs", 0.0),
        format="%.6f"
    )

    rap = st.number_input(
        "MDVP:RAP",
        value=st.session_state.get("rap", 0.0),
        format="%.6f"
    )

    ppq = st.number_input(
        "MDVP:PPQ",
        value=st.session_state.get("ppq", 0.0),
        format="%.6f"
    )

    ddp = st.number_input(
        "Jitter:DDP",
        value=st.session_state.get("ddp", 0.0),
        format="%.6f"
    )

    shimmer = st.number_input(
        "MDVP:Shimmer",
        value=st.session_state.get("shimmer", 0.0),
        format="%.6f"
    )

    shimmer_db = st.number_input(
        "MDVP:Shimmer(dB)",
        value=st.session_state.get("shimmer_db", 0.0),
        format="%.6f"
    )

    apq3 = st.number_input(
        "Shimmer:APQ3",
        value=st.session_state.get("apq3", 0.0),
        format="%.6f"
    )

with col2:

    apq5 = st.number_input(
        "Shimmer:APQ5",
        value=st.session_state.get("apq5", 0.0),
        format="%.6f"
    )

    apq = st.number_input(
        "MDVP:APQ",
        value=st.session_state.get("apq", 0.0),
        format="%.6f"
    )

    dda = st.number_input(
        "Shimmer:DDA",
        value=st.session_state.get("dda", 0.0),
        format="%.6f"
    )

    nhr = st.number_input(
        "NHR",
        value=st.session_state.get("nhr", 0.0),
        format="%.6f"
    )

    hnr = st.number_input(
        "HNR",
        value=st.session_state.get("hnr", 0.0),
        format="%.6f"
    )

    rpde = st.number_input(
        "RPDE",
        value=st.session_state.get("rpde", 0.0),
        format="%.6f"
    )

    dfa = st.number_input(
        "DFA",
        value=st.session_state.get("dfa", 0.0),
        format="%.6f"
    )

    spread1 = st.number_input(
        "spread1",
        value=st.session_state.get("spread1", 0.0),
        format="%.6f"
    )

    spread2 = st.number_input(
        "spread2",
        value=st.session_state.get("spread2", 0.0),
        format="%.6f"
    )

    d2 = st.number_input(
        "D2",
        value=st.session_state.get("d2", 0.0),
        format="%.6f"
    )

    ppe = st.number_input(
        "PPE",
        value=st.session_state.get("ppe", 0.0),
        format="%.6f"
    )

# ---------------- PREDICTION ----------------

if st.button("Predict Disease"):

    input_data = np.array([[
        fo, fhi, flo,
        jitter_percent,
        jitter_abs,
        rap,
        ppq,
        ddp,
        shimmer,
        shimmer_db,
        apq3,
        apq5,
        apq,
        dda,
        nhr,
        hnr,
        rpde,
        dfa,
        spread1,
        spread2,
        d2,
        ppe
    ]])

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)

    st.write("")

    if prediction[0] == 1:
        st.markdown(
            '<div class="error-box">⚠ Parkinson Disease Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="success-box">✅ Healthy Person</div>',
            unsafe_allow_html=True
        )
# ---------------- FOOTER ----------------        
st.markdown("---")
st.markdown(
    "<center>Developed by Rounak Kumar Gupta</center>",
    unsafe_allow_html=True
)