import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_and_save_model():
    """Train and save the machine learning model"""
    try:
        # Load dataset
        if not os.path.exists('train_crowd_data.csv'):
            raise FileNotFoundError("train_crowd_data.csv not found!")
        
        df = pd.read_csv('train_crowd_data.csv')
        
        # Encode categorical features
        le_source = LabelEncoder()
        le_destination = LabelEncoder()
        le_time = LabelEncoder()
        le_crowd = LabelEncoder()
        
        df['Source_enc'] = le_source.fit_transform(df['Source'])
        df['Destination_enc'] = le_destination.fit_transform(df['Destination'])
        df['Time_enc'] = le_time.fit_transform(df['Preferred_Time'])
        df['Crowd_Level_enc'] = le_crowd.fit_transform(df['Crowd_Level'])
        
        # Features & Target
        X = df[['Source_enc', 'Destination_enc', 'Time_enc']]
        y = df['Crowd_Level_enc']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create model directory
        os.makedirs('model', exist_ok=True)
        
        # Save model
        with open('model/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save encoders
        encoders = {
            'le_source': le_source,
            'le_destination': le_destination,
            'le_time': le_time,
            'le_crowd': le_crowd,
            'data': df  # Keep raw data for fetching prediction info
        }
        
        with open('model/encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
        
        print("✅ Model and encoders saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model()