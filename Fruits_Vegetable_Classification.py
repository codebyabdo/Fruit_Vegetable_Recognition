import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import numpy as np
import requests
from bs4 import BeautifulSoup
import os

# ---- Page Configuration ----
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS Styling ----
st.markdown("""
<style>
/* ==========================
   Global Image Styling
   ========================== */
.uploaded-img {
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    width: 180px;
    max-width: 100%;
    height: auto;
}

/* ==========================
   Main Title
   ========================== */
.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            border-radius: 15px;
    padding: 1rem;
}

/* ==========================
   Subtitle
   ========================== */
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 3rem;
}

/* ==========================
   Category Badges
   ========================== */
.category-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-weight: bold;
    font-size: 1rem;
    margin: 0.5rem 0;
}
.fruit-badge {
    background: #FF6B6B;
    color: white;
}
.vegetable-badge {
    background: #4ECDC4;
    color: white;
}

/* ==========================
   Prediction Text
   ========================== */
.prediction-text {
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    margin: 1rem 0;
}

/* ==========================
   Buttons
   ========================== */
.stButton>button {
    background: linear-gradient(45deg, #FF6B6B, #ee5a24);
    color: white !important;
    border: none;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    border-radius: 50px;
    transition: all 0.3s ease;
    width: 100%;
    cursor: pointer;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255,107,107,0.4);
}

/* ==========================
   Info Box
   ========================== */
.info-box {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
            
           img{
            border-radius: 15px !important;
            Width: 180px !important;
            
            }
            

/* ==========================
   Supported Items Cards
   ========================== */
.supported-items {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}
.supported-items:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

/* ==========================
   Result Cards
   ========================== */
.result-card {
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 5px solid;
    transition: transform 0.3s ease;
}
.result-card:hover {
    transform: translateY(-5px);
}
.fruit-card {
    border-left-color: #FF6B6B;
    background: linear-gradient(135deg, #fff5f5, #ffffff);
}
.vegetable-card {
    border-left-color: #4ECDC4;
    background: linear-gradient(135deg, #f0fff4, #ffffff);
}

/* ==========================
   Calorie Card
   ========================== */
.calorie-card {
    background: linear-gradient(135deg, #ffeaa7, #ffffff);
    border-left: 5px solid #fdcb6e;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* ==========================
   Responsive Adjustments
   ========================== */
@media (max-width: 768px) {
    .main-title { font-size: 2.5rem; }
    .prediction-text { font-size: 1.5rem; }
    .uploaded-img { width: 140px; }
}
</style>
""", unsafe_allow_html=True)


# ---- Ensure upload folder exists ----
os.makedirs("upload_images", exist_ok=True)

# ---- Load trained model with caching ----
@st.cache(allow_output_mutation=True)
def load_model_cached():
    try:
        model = load_model('FV.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None



model = load_model_cached()

# ---- Label dictionary ----
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn',
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

# ---- Category lists ----
fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi',
          'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple',
          'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn',
              'Cucumber', 'Eggplant', 'Ginger', 'Lettuce', 'Onion', 'Peas',
              'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn',
              'Sweetpotato', 'Tomato', 'Turnip']


# ---- Fetch calories from API ----
def fetch_calories(prediction):
    try:
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={prediction}&search_simple=1&action=process&json=1"
        response = requests.get(url).json()

        if "products" not in response or len(response["products"]) == 0:
            return "Calories not found"

        product = response["products"][0]

        # Look for calories per 100g
        if "nutriments" in product and "energy-kcal_100g" in product["nutriments"]:
            calories = product["nutriments"]["energy-kcal_100g"]
            return f"{calories} calories"
        else:
            return "Calories not found"

    except Exception as e:
        print("API Error:", e)
        return "Error fetching calories"


# ---- Image processing ----
def processed_img(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        y_class = prediction.argmax(axis=-1)[0]
        result = labels[y_class]

        return result.capitalize()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ---- Main App ----
def main():
    # Header Section
    st.markdown('<h1 class="main-title">🍎 Fruit & Vegetable Classifier 🥦</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image to identify fruits/vegetables and get nutritional information</p>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload Section
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown('### 📷 Upload Your Image')
        st.markdown('Supported formats: JPG, PNG, JPEG')
        img_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if img_file is not None:
            # Display uploaded image
            img = Image.open(img_file).resize((300, 300))
            st.image(img, caption="📸 Uploaded Image", use_column_width=True)
            
            # Save image
            save_path = f'upload_images/{img_file.name}'
            with open(save_path, "wb") as f:
                f.write(img_file.getbuffer())
            
            # Analyze button
            if st.button("🔍 Analyze Image", key="analyze"):
                with st.spinner('🔄 Analyzing image... Please wait'):
                    result = processed_img(save_path)
                    
                    if result:
                        # Display results
                        if result in vegetables:
                            card_class = "vegetable-card"
                            badge_class = "vegetable-badge"
                            badge_text = "🥬 VEGETABLE"
                        else:
                            card_class = "fruit-card"
                            badge_class = "fruit-badge"
                            badge_text = "🍎 FRUIT"
                        
                        st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
                        st.markdown(f'<div class="category-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="prediction-text">🎯 {result}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Fetch and display calories
                        with st.spinner('📊 Fetching nutritional information...'):
                            cal = fetch_calories(result)
                        
                        st.markdown('<div class="calorie-card">', unsafe_allow_html=True)
                        st.markdown('### 🔥 Nutritional Information')
                        st.markdown(f'### {cal} (per 100g)')
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
    # Information Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                    padding: 2rem;
                    border-radius: 20px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                    margin-bottom: 2rem;">
            <h3 style="color: #fff;">🎯 How It Works</h3>
            <p>This AI-powered classifier uses deep learning to identify fruits and vegetables from images.</p>

            <h4 style="color: #FF6B35;">📊 Supported Items</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                <div style="background: rgba(255,255,255,0.85); padding: 1rem; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    <h5>🍎 Fruits (15 types)</h5>
                    <small>Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Watermelon, and more...</small>
                </div>
                <div style="background: rgba(255,255,255,0.85); padding: 1rem; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    <h5>🥦 Vegetables (20 types)</h5>
                    <small>Carrot, Tomato, Potato, Onion, Cabbage, Cucumber, Bell Pepper, Spinach, and more...</small>
                </div>
            </div>

            <h4 style="color: #FF69B4; margin-top: 20px;">💡 Tips for Best Results</h4>
            <ul style="margin-left: 20px;">
                <li>Use clear, well-lit images</li>
                <li>Focus on a single item</li>
                <li>Avoid blurry photos</li>
                <li>Plain background works best</li>
                <li>Close-up shots recommended</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Statistics Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 1.5rem;
                    border-radius: 20px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                    margin-top: 2rem;">
            <h4 style="color: #fffacd;">📈 Model Statistics</h4>
            <p><strong>36</strong> different food items</p>
            <p><strong>15</strong> fruit varieties</p>
            <p><strong>20</strong> vegetable varieties</p>
            <p><strong>AI-powered</strong> classification</p>
            <p><strong>Real-time</strong> calorie estimation</p>
        </div>
        """, unsafe_allow_html=True)



# Run the app
if __name__ == "__main__":
    main()