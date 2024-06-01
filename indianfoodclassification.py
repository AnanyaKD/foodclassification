import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import plt

# Load YOLOv8 model (assumes the model is pretrained for Indian food classification)
model_path = "/Users/anushkadurg/Documents/FINAL!/best.pt"

# Streamlit app configuration
st.set_page_config(page_title="Food Classification App", page_icon="🍲", layout="centered")
st.title("🍽️ Food Classification App")

# Custom background for the app
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://tint.creativemarket.com/_tdX-cL_1LzfGtnO_ks3o0PJwJvEKgKQcTFa78xF06U/width:3640/height:2410/gravity:nowe/rt:fill-down/el:1/preset:cm_watermark_retina/czM6Ly9maWxlcy5jcmVhdGl2ZW1hcmtldC5jb20vaW1hZ2VzL3NjcmVlbnNob3RzL3Byb2R1Y3RzLzEyMzMvMTIzMzcvMTIzMzc4NDYvMy0wOTcxNi1vLmpwZw");
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model loading function to avoid reloading on every run
@st.cache_resource
def load_model():
    model = YOLO(model_path)
    return model

# Placeholder for storing reviews
if 'reviews' not in st.session_state:
    st.session_state.reviews = []

# Create three tabs for different sections
tab1, tab2, tab3 = st.tabs(["Image Classification", "About", "User Reviews"])

with tab1:
    st.header("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        model = load_model()

        with st.spinner("Classifying... Please wait..."):
            # Perform detection directly with PIL image
            results = model.predict(image)
            top5_indices = results[0].probs.top5
            top5_confidences = results[0].probs.top5conf.numpy()

        st.header("Top 5 Predictions:")
        predictions = []
        probabilities = []
        for idx, ind in enumerate(top5_indices):
            dish_name = results[0].names[ind]
            prob = top5_confidences[idx]
            predictions.append(dish_name)
            probabilities.append(prob)
            st.subheader(f"🔸 {dish_name} ({prob:.2%})")

        # Display the probabilities as a bar chart
        fig, ax = plt.subplots()
        ax.barh(predictions, probabilities, color='skyblue')
        ax.set_xlabel('Probability')
        ax.set_title('Top 5 Predictions')
        st.pyplot(fig)

            
# Tab 2: About
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

# Tab 3: User Reviews
with tab3:
    st.header("User Reviews")

    # Add a form to submit reviews
    with st.form(key='user_review_form'):
        review_text = st.text_input("Leave a review:")
        review_rating = st.slider("Rate the dish:", 1, 5)
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            review = {
                "text": review_text,
                "rating": review_rating
            }
            st.session_state.reviews.append(review)
            st.success("Review submitted!")

    # Display all reviews
    if st.session_state.reviews:
        st.header("User Reviews")
        for review in st.session_state.reviews:
            st.write(f"Rating: {review['rating']} ⭐")
            st.write(review['text'])
