import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model (assumes the model is pretrained for Indian food classification)
model_path = "/Users/anushkadurg/Documents/FINAL!"

# Streamlit app
st.set_page_config(page_title="Food Classification App", page_icon="🍲", layout="centered")
st.title("🍽️ Food Classification App")

st.markdown("""
                <style>
    .stApp {
    background-image: url("https://as2.ftcdn.net/v2/jpg/05/64/97/31/1000_F_564973106_xO0n1BvNmsFy2Sj2SK2Uy2pzH3hDqeDy.jpg");
    background-size: cover;
    }
    </style>
                """, unsafe_allow_html = True)



@st.cache_resource
def load_model():
    mod=YOLO("/Users/anushkadurg/Documents/runs/classify/train/weights/best.pt ")
    return mod
    #return YOLO(model_path)


# Create two tabs
tab1, tab2 = st.tabs(["Image Classification", "About"])

# Tab 1: Image Classification
with tab1:
    st.header("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Classifying...")
        mod1 = load_model()

        # Perform detection directly with PIL image
        results = mod1.predict(image)
        top5_preds = results[0].probs.top5

        st.header("Top 5 Predictions:")
        for ind in top5_preds:
            st.header(f"🔸 {results[0].names[ind]}")
        
##Tab 2: About
with tab2:
    st.header("About this App")

    st.markdown("""
    <div style="background-color: #e0f7fa; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
        <h4>Purpose of this App</h4>
        <p>
        This Food Classification app allows users to upload images of food dishes, 
        and the app will classify the items using a pretrained YOLOv8 model. 
        The purpose of this app is to help users quickly identify various dishes from 
        an image, making it easier to recognize and learn about different foods.
        </p>
    </div>
    <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
        <h4>Features</h4>
        <ul>
            <li>🍛 <b>Upload</b> an image of a food dish.</li>
            <li>🤖 <b>Detects</b> the dish using YOLO technology.</li>
            <li>📊 <b>Get</b> the top 5 predictions with probabilities.</li>
            <li>🧠 <b>Learn</b> about different dishes around the world.</li>
        </ul>
    </div>
    <div style="background-color: #e6ee9c; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
        <h4>How to Use</h4>
        <ol>
            <li>Go to the <b>Image Classification</b> tab.</li>
            <li><b>Upload</b> an image of a dish.</li>
            <li>Wait for the <b>detection results</b>.</li>
            <li><b>Explore</b> the predictions and learn about new dishes!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
