import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os
import re

# Set page config
st.set_page_config(
    page_title="Train Crowd Predictor",
    page_icon="🚆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 2rem;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #b3d9ff;
        border-radius: 12px;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #007bff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    .train-header {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .info-item {
        background: #f8f9fa;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    
    .crowd-high { color: #dc3545; background: #f8d7da; }
    .crowd-medium { color: #ffc107; background: #fff3cd; }
    .crowd-low { color: #28a745; background: #d4edda; }
    
    .swap-button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Train name mapping
TRAIN_NAMES = {
    "12723": "Godavari Express",
    "12711": "Simhadri Express",
    "12733": "Narasimha Express",
    "12761": "Padmavati Express",
    "12762": "Visakha Express",
    "12759": "Charminar Express",
    "12578": "Bagmati Express",
    "12603": "Hyderabad Express",
    "12712": "Pinakini Express",
    "12705": "Hussainsagar Express",
    "12679": "Cocanada Express",
    "12798": "Venkatadri Express",
    "12805": "Janmabhoomi Express",
    "12863": "Howrah Express",
    "12754": "Nagarjuna Express",
    "12737": "Goutami Express",
    "12669": "Gangavaram Express",
    "12604": "Chennai Express",
    "12786": "Tirumala Express",
    "12713": "Satavahana Express",
    "12706": "Amaravati Express",
    "12680": "Intercity Express",
    "12799": "Rayalaseema Express",
    "12806": "East Coast Express",
    "12864": "Coromandel Express",
    "12755": "Krishna Express",
    "12738": "Godavari Express",
    "12610": "Chennai Express",
    "12714": "Sabari Express",
    "12760": "Charminar Express",
    "12616": "GT Express",
    "12539": "YPR Express",
    "12295": "Sanghamitra Express"
}

def get_train_name(train_number):
    """Extract train name from train number"""
    if not train_number or train_number == 'N/A' or pd.isna(train_number):
        return None
    
    # Extract number part
    number_match = re.search(r'\d+', str(train_number))
    if number_match:
        number = number_match.group()
        train_name = TRAIN_NAMES.get(number)
        if train_name:
            return train_name
    
    # Fallback based on number pattern
    if number_match:
        number = number_match.group()
        if number.startswith('12'):
            return "Express Train"
        elif number.startswith('22'):
            return "AC Express"
        elif number.startswith('18'):
            return "Mail Express"
        else:
            return "Passenger Train"
    
    return None

# Check if model exists, if not train it
@st.cache_resource
def load_model_and_data():
    if not os.path.exists('model/model.pkl'):
        with st.spinner('Training model for the first time...'):
            from train_model import train_and_save_model
            train_and_save_model()
    
    # Load model and encoders
    model = pickle.load(open("model/model.pkl", "rb"))
    encoders = pickle.load(open("model/encoders.pkl", "rb"))
    return model, encoders

# Load model and data
try:
    model, encoders = load_model_and_data()
    df = encoders['data']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🚆 Train Crowd Predictor</h1>
    <p style="color: #666; font-size: 1.1rem;">Find the best train with crowd analysis and schedule optimization</p>
</div>
""", unsafe_allow_html=True)

# Main form
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Form inputs
        stations = sorted(df['Source'].unique().tolist())
        times = sorted(df['Preferred_Time'].unique().tolist())
        
        # Source station
        source = st.selectbox(
            "🗺️ Source Station",
            options=[""] + stations,
            index=0,
            help="Select your departure station"
        )
        
        # Destination station with swap functionality
        col_dest, col_swap = st.columns([5, 1])
        with col_dest:
            destination = st.selectbox(
                "🎯 Destination Station",
                options=[""] + stations,
                index=0,
                help="Select your arrival station"
            )
        
        with col_swap:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⇄", help="Swap stations", key="swap_btn"):
                if source and destination:
                    # Store in session state for swapping
                    st.session_state.temp_source = destination
                    st.session_state.temp_destination = source
        
        # Apply swapped values if they exist
        if 'temp_source' in st.session_state:
            source = st.session_state.temp_source
            destination = st.session_state.temp_destination
            # Clear temporary values
            del st.session_state.temp_source
            del st.session_state.temp_destination
            st.experimental_rerun()
        
        # Preferred time
        preferred_time = st.selectbox(
            "🕒 Preferred Time",
            options=[""] + times,
            index=0,
            help="Select your preferred travel time"
        )
        
        # Predict button
        predict_button = st.button(
            "🎯 Find Best Train",
            type="primary",
            use_container_width=True
        )

# Prediction logic
if predict_button:
    if not source or not destination or not preferred_time:
        st.error("Please select all fields: Source, Destination, and Preferred Time")
    elif source == destination:
        st.warning("Source and destination cannot be the same!")
    else:
        try:
            with st.spinner('🔍 Finding the best train for you...'):
                # Encode inputs
                src_enc = encoders['le_source'].transform([source])[0]
                dst_enc = encoders['le_destination'].transform([destination])[0]
                time_enc = encoders['le_time'].transform([preferred_time])[0]
                
                # Predict crowd level
                prediction = model.predict([[src_enc, dst_enc, time_enc]])[0]
                crowd_label = encoders['le_crowd'].inverse_transform([prediction])[0]
                
                # Filter from original dataset
                filtered = df[(df['Source'] == source) &
                             (df['Destination'] == destination) &
                             (df['Preferred_Time'] == preferred_time)]
                
                if not filtered.empty:
                    match = filtered.iloc[0]
                    main_train = match['Best_Train']
                    alt_train = match['Alternate_Train']
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🚆 Your Suggested Train")
                    
                    # Main result card
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="train-header">
                                <h3>{main_train}</h3>
                                {f"<small>{get_train_name(main_train)}</small>" if get_train_name(main_train) else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Information in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="info-item">
                                <strong>🕒 Departure:</strong><br>
                                {match['Departure']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="info-item">
                                <strong>📈 Crowd Level:</strong><br>
                                <span class="crowd-{crowd_label.lower()}">{crowd_label}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="info-item">
                                <strong>⏳ Standing Time:</strong><br>
                                {match['Expected_Standing_Time']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="info-item">
                                <strong>🧭 Arrival:</strong><br>
                                {match['Arrival']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="info-item">
                                <strong>🪑 Seat Available After:</strong><br>
                                {match['Seat_Likely_Available_After']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Alternative suggestion
                    st.markdown("### 💡 Alternative Option")
                    alt_train_name = get_train_name(alt_train)
                    st.info(f"**Suggested alternative train:** {alt_train}" + 
                           (f" ({alt_train_name})" if alt_train_name else ""))
                    
                    # Notice
                    st.markdown("---")
                    st.info("🔔 This prediction is only for General Compartments in Indian Railways.")
                    
                    # Feedback after 3 seconds
                    if st.button("👍 Was this prediction accurate?"):
                        st.success("Thank you for your feedback! 🙏")
                else:
                    st.warning("No direct train found for this route and time. Please try different options.")
                    
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please try again or contact support if the issue persists.")

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 📊 About")
    st.write("""
    This app predicts train crowd levels and suggests the best trains 
    for your journey in Andhra Pradesh.
    """)
    
    st.markdown("## 🎯 Features")
    st.write("""
    - Real-time crowd prediction
    - Best train suggestions
    - Alternative options
    - Seat availability info
    - Standing time estimates
    """)
    
    st.markdown("## 📍 Coverage")
    st.write("""
    Major stations covered:
    - Vijayawada, Guntur, Tirupati
    - Visakhapatnam, Nellore
    - Hyderabad, Warangal
    - And many more...
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Made with ❤️ for Indian Railway Passengers<br>
    <small>Powered by Machine Learning & Streamlit</small>
</div>
""", unsafe_allow_html=True)