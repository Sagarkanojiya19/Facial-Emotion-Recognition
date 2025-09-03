# 🎭 Enhanced Facial Emotion Detection App with Professional UI
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import time

def load_model():
    """Load model with TensorFlow compatibility fixes"""
    model_path = 'emotion_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return None
    
    try:
        print(f"📂 Loading model from: {model_path}")
        
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
        }
        
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=custom_objects
        )
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e1:
        print(f"⚠️ Method 1 failed: {e1}")
        
        try:
            print("🔄 Trying alternative loading method...")
            
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            
            try:
                model.load_weights(model_path)
                print("✅ Model weights loaded successfully!")
                return model
            except:
                print("⚠️ Could not load weights, using random initialization")
                return model
                
        except Exception as e2:
            print(f"❌ Method 2 also failed: {e2}")
            return None

# Emotion classes with emojis
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢', 
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

emotion_descriptions = {
    'angry': 'Anger - Strong feeling of displeasure',
    'disgust': 'Disgust - Feeling of strong disapproval', 
    'fear': 'Fear - Feeling of being afraid or worried',
    'happy': 'Happiness - Feeling of joy and contentment',
    'neutral': 'Neutral - No particular emotion expressed',
    'sad': 'Sadness - Feeling of sorrow or unhappiness',
    'surprise': 'Surprise - Feeling of being astonished'
}

# Cache the model
model_cache = None
def get_model():
    global model_cache
    if model_cache is None:
        model_cache = load_model()
    return model_cache

def preprocess_image(image):
    """Preprocess image for emotion detection"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = cv2.resize(image, (48, 48))
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 48, 48, 1)
    
    return image
    
def predict_emotion(image):
    """Predict emotion from image with detailed results"""
    try:
        model = get_model()
        if model is None:
            return "Model loading failed", 0.0, {}, "Please check model file", "⚠️"
        
        processed_image = preprocess_image(np.array(image))
        
        # Add small delay for better UX
        time.sleep(0.5)
        
        predictions = model.predict(processed_image, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        emotion = class_names[predicted_class]
        
        results = {}
        for i, class_name in enumerate(class_names):
            results[class_name] = float(predictions[0][i])
        
        # Get emoji and description
        emoji = emotion_emojis.get(emotion, '🤔')
        description = emotion_descriptions.get(emotion, 'Unknown emotion')
        
        return emotion, confidence, results, description, emoji
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return f"Error: {str(e)}", 0.0, {}, "Prediction failed", "❌"

def format_prediction_result(emotion, confidence, description, emoji):
    """Format the prediction result with dark mode styled card"""
    if "Error" in emotion or "failed" in emotion:
        return f"""
        <div style="
            position: relative;
            padding: 40px 32px;
            background: linear-gradient(145deg, #1a202c, #2d3748);
            border-radius: 12px;
            border-left: 4px solid #ef4444;
            color: #f9fafb;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-top: 24px;
            text-align: center;
            overflow: hidden;
        ">
            <div style="font-size: 48px; margin-bottom: 24px;">⚠️</div>
            <div style="font-size: 24px; font-weight: 600; margin-bottom: 12px; color: #f9fafb;">Detection Failed</div>
            <div style="font-size: 16px; color: #e5e7eb; margin-bottom: 24px;">{emotion}</div>
            
            <!-- Decorative status dot -->
            <div style="
                position: absolute;
                top: 20px;
                right: 20px;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #ef4444;
            "></div>
        </div>
        """
    
    # Confidence color and badge style based on score
    if confidence >= 70:
        confidence_color = "#22c55e"  # Green
        confidence_text = "High Confidence"
        confidence_bg = "rgba(34, 197, 94, 0.1)"
        confidence_border = "rgba(34, 197, 94, 0.2)"
    elif confidence >= 50:
        confidence_color = "#f59e0b"  # Amber
        confidence_text = "Medium Confidence"
        confidence_bg = "rgba(245, 158, 11, 0.1)"
        confidence_border = "rgba(245, 158, 11, 0.2)"
    else:
        confidence_color = "#ef4444"  # Red
        confidence_text = "Low Confidence"
        confidence_bg = "rgba(239, 68, 68, 0.1)"
        confidence_border = "rgba(239, 68, 68, 0.2)"
    
    # Get emotion-specific color
    emotion_colors = {
        'angry': '#ef4444',
        'disgust': '#10b981',
        'fear': '#8b5cf6',
        'happy': '#f59e0b',
        'neutral': '#6b7280',
        'sad': '#3b82f6',
        'surprise': '#ec4899'
    }
    emotion_color = emotion_colors.get(emotion.lower(), '#6366f1')
    
    return f"""
    <div style="
        position: relative;
        padding: 40px 32px;
        background: linear-gradient(145deg, #1a202c, #2d3748);
        border-radius: 12px;
        border-left: 4px solid {emotion_color};
        color: #f9fafb;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-top: 24px;
        text-align: center;
        overflow: hidden;
    ">
        <!-- Emotion emoji -->
        <div style="font-size: 72px; margin-bottom: 24px;">{emoji}</div>
        
        <!-- Emotion name -->
        <div style="font-size: 28px; font-weight: 700; margin-bottom: 12px; color: #f9fafb;">
            {emotion.capitalize()}
        </div>
        
        <!-- Description -->
        <div style="font-size: 16px; color: #e5e7eb; margin-bottom: 24px; max-width: 480px; margin-left: auto; margin-right: auto;">
            {description}
        </div>
        
        <!-- Decorative status dot -->
        <div style="
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: {emotion_color};
        "></div>
        
        <!-- Background decoration element -->
        <div style="
            position: absolute;
            top: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 30px;
            background: {emotion_color};
            opacity: 0.1;
            z-index: 0;
        "></div>
        
        <!-- Background decoration element -->
        <div style="
            position: absolute;
            bottom: 20px;
            left: 20px;
            width: 40px;
            height: 40px;
            border-radius: 20px;
            background: {emotion_color};
            opacity: 0.05;
            z-index: 0;
        "></div>
    </div>
    """

def predict_interface(image):
    """Modern minimal prediction interface with single output"""
    if image is None:
        return """
        <div style="
            text-align: center;
            padding: 48px 24px;
            background: #1f2937;
            border-radius: 12px;
            color: #f9fafb;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(75, 85, 99, 0.4);
        ">
            <div style="
                width: 80px;
                height: 80px;
                margin: 0 auto 24px auto;
                background: #374151;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 16C9.85038 16.6303 10.8846 17 12 17C13.1154 17 14.1496 16.6303 15 16" stroke="#9ca3af" stroke-width="1.5" stroke-linecap="round"/>
                    <path d="M12 13V13.01" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"/>
                    <path d="M15 8H15.01" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"/>
                    <path d="M9 8H9.01" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#9ca3af" stroke-width="1.5"/>
                </svg>
            </div>
            <h3 style="margin: 0 0 8px 0; font-size: 20px; font-weight: 600; color: #f9fafb;">Ready to Analyze</h3>
            <p style="margin: 0; font-size: 16px; color: #9ca3af;">Upload an image or take a photo to detect emotions</p>
        </div>
        """
    
    emotion, confidence, all_results, description, emoji = predict_emotion(image)
    
    # Format result using our new modern minimal style
    result_html = format_prediction_result(emotion, confidence, description, emoji)
    
    return result_html

# Complete App Redesign - Dark Mode Professional UI
custom_css = """
/* Base Setup and Variables */
:root {
    --primary-color: #6366f1;
    --primary-dark: #4f46e5;
    --primary-light: #818cf8;
    --secondary-color: #ec4899;
    --bg-color: #111827;
    --bg-card: #1f2937;
    --bg-sidebar: #1e293b;
    --text-primary: #f9fafb;
    --text-secondary: #e5e7eb;
    --text-muted: #9ca3af;
    --border-color: rgba(75, 85, 99, 0.4);
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    --radius-sm: 0.375rem;
    --radius: 0.5rem;
    --radius-md: 0.75rem;
    --radius-lg: 1rem;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    --transition: all 0.2s ease;
}

/* Reset and Base Styles */
.gradio-container {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-color) !important;
    line-height: 1.5 !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container * {
    box-sizing: border-box;
}

/* Main Layout */
.gradio-container > .main-app > .gr-interface {
    background-color: var(--bg-color);
    padding: 0;
    margin: 0;
    border: none;
    box-shadow: none;
}

/* Sidebar Panel */
.sidebar-panel {
    background-color: var(--bg-sidebar) !important;
    border-radius: var(--radius) !important;
    height: 100% !important;
    padding: 0 !important;
    margin-right: 16px !important;
    max-width: 320px !important;
    position: relative !important;
    box-shadow: var(--shadow-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-color) !important;
}

.sidebar-container {
    display: flex;
    flex-direction: column;
    padding: 24px 16px;
    gap: 24px;
    height: 100%;
    color: var(--text-primary);
}

/* App Logo */
.app-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.logo-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    padding: 8px;
}

.logo-text {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(to right, var(--primary-light), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* App Description */
.app-description {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.5;
    opacity: 0.9;
    margin-bottom: 24px;
    padding-left: 2px;
}

/* Emotion Indicator Grid */
.emotion-indicator {
    background-color: rgba(31, 41, 55, 0.5);
    border-radius: var(--radius);
    padding: 16px;
    border: 1px solid var(--border-color);
}

.indicator-title {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.emotion-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
    gap: 8px;
}

.emotion-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px;
    border-radius: var(--radius-sm);
    transition: var(--transition);
    border: 1px solid transparent;
    cursor: default;
}

.emotion-item:hover {
    background-color: rgba(255, 255, 255, 0.05);
    transform: translateY(-2px);
    border-color: var(--primary-color);
}

.emotion-emoji {
    font-size: 1.5rem;
    margin-bottom: 4px;
}

.emotion-name {
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* Status Indicator */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: rgba(31, 41, 55, 0.3);
    padding: 8px 12px;
    border-radius: 50px;
    width: fit-content;
    margin-top: auto;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    position: relative;
}

.status-dot::after {
    content: "";
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border-radius: 50%;
    background-color: rgba(16, 185, 129, 0.3);
    animation: pulse 2s ease-in-out infinite;
}

.status-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* Sidebar Tips */
.sidebar-tips {
    background-color: rgba(31, 41, 55, 0.5);
    border-radius: var(--radius);
    padding: 16px;
    border: 1px solid var(--border-color);
    margin-top: 24px;
}

.tips-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 12px;
}

.tips-list {
    list-style: none;
    padding: 0;
    margin: 0;
    color: var(--text-muted);
    font-size: 0.8rem;
}

.tips-list li {
    margin-bottom: 8px;
    padding-left: 18px;
    position: relative;
    line-height: 1.5;
}

.tips-list li::before {
    content: "•";
    color: var(--primary-light);
    position: absolute;
    left: 0;
    font-weight: bold;
}

/* Sidebar Credits */
.sidebar-credits {
    margin-top: 24px;
}

.credits-title {
    color: var(--text-secondary);
    font-size: 0.75rem;
    margin-bottom: 12px;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.tech-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 50px;
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* Content Boxes */
.content-box {
    background: linear-gradient(145deg, #1a202c, #2d3748) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    box-shadow: var(--shadow-md) !important;
    margin-bottom: 16px !important;
    border: 1px solid var(--border-color) !important;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
}

.section-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background-color: rgba(99, 102, 241, 0.1);
    border-radius: 12px;
    color: var(--primary-color);
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
}

.upload-description {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* Image Component */
.image-input-component {
    margin: 0 auto !important;
    max-width: 420px !important;
}

.gradio-container .image-input-component .image-container {
    background-color: rgba(31, 41, 55, 0.5) !important;
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
    padding: 20px !important;
}

.gradio-container .image-input-component .image-container:hover {
    border-color: var(--primary-color) !important;
    background-color: rgba(99, 102, 241, 0.05) !important;
}

/* Buttons */
.action-buttons {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 20px;
}

.gradio-container .analyze-button {
    background: linear-gradient(135deg, var(--primary-color), #4f46e5) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    border-radius: var(--radius) !important;
    border: none !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4), 0 2px 4px -1px rgba(99, 102, 241, 0.2) !important;
}

.gradio-container .analyze-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px -3px rgba(99, 102, 241, 0.4), 0 4px 6px -2px rgba(99, 102, 241, 0.2) !important;
    filter: brightness(1.05) !important;
}

.gradio-container .clear-button {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-color) !important;
    transition: all 0.2s ease !important;
}

.gradio-container .clear-button:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* Result Display */
.result-display-component {
    min-height: 300px !important;
    display: flex !important;
    align-items: stretch !important;
    justify-content: center !important;
}

.initial-result-state {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    width: 100%;
}

.result-placeholder {
    text-align: center;
    padding: 32px;
    background-color: rgba(31, 41, 55, 0.3);
    border-radius: var(--radius);
    max-width: 300px;
}

.result-placeholder-icon {
    margin: 0 auto 20px;
    width: 80px;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 50%;
    color: var(--text-muted);
}

.result-placeholder-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.result-placeholder-description {
    font-size: 0.9rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* Emotion Result Card */
.emotion-result-card {
    background-color: var(--bg-card);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
}

.emotion-result-emoji {
    font-size: 64px;
    margin-bottom: 24px;
}

.emotion-result-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.025em;
}

.emotion-result-description {
    font-size: 16px;
    color: var(--text-secondary);
    margin-bottom: 24px;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
}

.emotion-result-confidence {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 50px;
    font-size: 14px;
    font-weight: 500;
}

.confidence-high {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.confidence-medium {
    background-color: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.2);
}

.confidence-low {
    background-color: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* Animation */
@keyframes pulse {
    0% {
        opacity: 0.6;
        transform: scale(1);
    }
    50% {
        opacity: 0.3;
        transform: scale(1.3);
    }
    100% {
        opacity: 0.6;
        transform: scale(1);
    }
}

@keyframes slideInUp {
    from {
        transform: translateY(20px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.animate-in {
    animation: slideInUp 0.3s ease-out forwards;
}

/* Overrides for Gradio Components */
.gradio-container [class*="message"] {
    color: var(--text-primary) !important;
    background-color: var(--bg-card) !important;
}

.gradio-container [class*="uploadButton"] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

.gradio-container [class*="wrap"] {
    background-color: transparent !important;
}

.gradio-container [class*="panel"] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

.gradio-container [class*="block"],
.gradio-container [class*="form"],
.gradio-container [class*="accordion"],
.gradio-container [class*="tabitem"] {
    background-color: transparent !important;
    color: var(--text-primary) !important;
}

.gradio-container [class*="label-wrap"],
.gradio-container [class*="block-title"],
.gradio-container [class*="block-label"],
.gradio-container [class*="label"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

.gradio-container [class*="header"] {
    background-color: var(--bg-sidebar) !important;
    color: var(--text-primary) !important;
}

/* Fix for built-in components */
.gradio-container .file-preview,
.gradio-container .file-preview * {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* Media queries */
@media (max-width: 768px) {
    .gradio-container .gr-row {
        flex-direction: column !important;
    }

    .sidebar-panel {
        margin-right: 0 !important;
        margin-bottom: 16px !important;
        max-width: 100% !important;
    }

    .emotion-grid {
        grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
    }
}

/* Atom One Dark Syntax Highlighting for any code blocks */
.hljs {
    display: block;
    overflow-x: auto;
    padding: 0.5em;
    color: #abb2bf;
    background: #282c34;
}
.hljs-comment, .hljs-quote {
    color: #5c6370;
    font-style: italic;
}
.hljs-doctag, .hljs-keyword, .hljs-formula {
    color: #c678dd;
}
.hljs-section, .hljs-name, .hljs-selector-tag, .hljs-deletion, .hljs-subst {
    color: #e06c75;
}
.hljs-literal {
    color: #56b6c2;
}
.hljs-string, .hljs-regexp, .hljs-addition, .hljs-attribute, .hljs-meta-string {
    color: #98c379;
}
.hljs-built_in, .hljs-class .hljs-title {
    color: #e6c07b;
}
.hljs-attr, .hljs-variable, .hljs-template-variable, .hljs-type, .hljs-selector-class, .hljs-selector-attr, .hljs-selector-pseudo, .hljs-number {
    color: #d19a66;
}
.hljs-symbol, .hljs-bullet, .hljs-link, .hljs-meta, .hljs-selector-id, .hljs-title {
    color: #61aeee;
}
.hljs-emphasis {
    font-style: italic;
}
.hljs-strong {
    font-weight: bold;
}
.hljs-link {
    text-decoration: underline;
}
"""




# Create the modern minimal interface
with gr.Blocks(
    title="Expression AI", 
    theme=gr.themes.Base(),
    css=custom_css
) as demo:
    
    # Modern Elegant Header Section
    gr.HTML("""
    <div style="
        background: radial-gradient(circle at top right, #4f46e5, #1e40af);
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        margin: 20px 0 40px 0;
        box-shadow: 0 20px 40px -10px rgba(30, 64, 175, 0.4);
    ">
        <!-- Decorative geometric elements -->
        <div style="
            position: absolute;
            top: -100px;
            left: -100px;
            width: 300px;
            height: 300px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.03);
            z-index: 1;
        "></div>
        
        <div style="
            position: absolute;
            bottom: -80px;
            right: -80px;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.05);
            z-index: 1;
        "></div>
        
        <div style="
            position: absolute;
            top: 40px;
            right: 20%;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            z-index: 1;
        "></div>
        
        <div style="
            position: absolute;
            bottom: 30px;
            left: 15%;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.15);
            z-index: 1;
        "></div>
        
        <!-- Content container -->
        <div style="
            position: relative;
            z-index: 2;
            padding: 60px 30px;
            text-align: center;
        ">
            <!-- Main Title with animated gradient -->
            <div style="
                font-size: 3.5rem;
                font-weight: 900;
                margin: 0 0 20px 0;
                background: linear-gradient(90deg, #fff, #e0e7ff);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                display: inline-block;
                letter-spacing: -0.025em;
                line-height: 1.1;
                text-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            ">
                Expression AI
            </div>
            
            <!-- Subtitle with enhanced styling -->
            <div style="
                font-size: 1.25rem;
                font-weight: 400;
                color: rgba(255, 255, 255, 0.9);
                margin: 0;
                line-height: 1.6;
                max-width: 650px;
                margin-left: auto;
                margin-right: auto;
                padding: 0 20px;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            ">
                Understand facial expressions through advanced expression recognition technology
            </div>
            
            <!-- Status Indicator with improved design -->
            <div style="
                position: absolute;
                top: 20px;
                right: 20px;
                display: flex;
                align-items: center;
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(12px);
                border-radius: 30px;
                padding: 6px 14px;
                gap: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            ">
                <div style="
                    width: 10px;
                    height: 10px;
                    background: #22c55e;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                "></div>
                <span style="font-size: 0.8rem; font-weight: 600; color: white; letter-spacing: 0.02em;">Ready</span>
            </div>
        </div>
    </div>
    
    <!-- CSS animations and styles -->
    <style>
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
        }
        70% {
            box-shadow: 0 0 0 8px rgba(34, 197, 94, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
        }
    }
    
    /* Enhanced emotion tag styling */
    .emotion-tag {
        padding: 8px 14px;
        border-radius: 30px;
        font-size: 0.875rem;
        font-weight: 700;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
    }
    
    .emotion-tag:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
    """)

    
    # Completely New Split Panel UI Layout
    with gr.Blocks() as main_interface:
        with gr.Row(equal_height=False):
            # Left Panel - Full Height Sidebar
            with gr.Column(scale=1, min_width=320, elem_classes="sidebar-panel"):
                gr.HTML("""
                <div class="sidebar-container">
                    <!-- App Logo -->
                    <div class="app-logo">
                        <div class="logo-icon">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M9 22H15C20 22 22 20 22 15V9C22 4 20 2 15 2H9C4 2 2 4 2 9V15C2 20 4 22 9 22Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M15.5 9.75C16.3284 9.75 17 9.07843 17 8.25C17 7.42157 16.3284 6.75 15.5 6.75C14.6716 6.75 14 7.42157 14 8.25C14 9.07843 14.6716 9.75 15.5 9.75Z" stroke="currentColor" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M8.5 9.75C9.32843 9.75 10 9.07843 10 8.25C10 7.42157 9.32843 6.75 8.5 6.75C7.67157 6.75 7 7.42157 7 8.25C7 9.07843 7.67157 9.75 8.5 9.75Z" stroke="currentColor" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M8.4 13.3H15.6C16.1 13.3 16.5 13.7 16.5 14.2C16.5 16.69 14.49 18.7 12 18.7C9.51 18.7 7.5 16.69 7.5 14.2C7.5 13.7 7.9 13.3 8.4 13.3Z" stroke="currentColor" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="logo-text">Expression AI</div>
                    </div>
                    
                    <!-- App Description -->
                    <div class="app-description">
                        Advanced facial expression analysis powered by machine learning
                    </div>
                    
                    <!-- Emotion Indicator -->
                    <div class="emotion-indicator">
                        <div class="indicator-title">Detectable Emotions</div>
                        <div class="emotion-grid">
                            <div class="emotion-item" data-emotion="angry">
                                <div class="emotion-emoji">😠</div>
                                <div class="emotion-name">Angry</div>
                            </div>
                            <div class="emotion-item" data-emotion="disgust">
                                <div class="emotion-emoji">🤢</div>
                                <div class="emotion-name">Disgust</div>
                            </div>
                            <div class="emotion-item" data-emotion="fear">
                                <div class="emotion-emoji">😨</div>
                                <div class="emotion-name">Fear</div>
                            </div>
                            <div class="emotion-item" data-emotion="happy">
                                <div class="emotion-emoji">😊</div>
                                <div class="emotion-name">Happy</div>
                            </div>
                            <div class="emotion-item" data-emotion="neutral">
                                <div class="emotion-emoji">😐</div>
                                <div class="emotion-name">Neutral</div>
                            </div>
                            <div class="emotion-item" data-emotion="sad">
                                <div class="emotion-emoji">😢</div>
                                <div class="emotion-name">Sad</div>
                            </div>
                            <div class="emotion-item" data-emotion="surprise">
                                <div class="emotion-emoji">😲</div>
                                <div class="emotion-name">Surprise</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Status Indicator -->
                    <div class="status-indicator">
                        <span class="status-dot"></span>
                        <span class="status-text">System Ready</span>
                    </div>
                </div>
                """)
                
                # Tips Section in Sidebar
                gr.HTML("""
                <div class="sidebar-tips">
                    <div class="tips-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 22C17.5 22 22 17.5 22 12C22 6.5 17.5 2 12 2C6.5 2 2 6.5 2 12C2 17.5 6.5 22 12 22Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M12 8V13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M11.9945 16H12.0035" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Tips for Best Results
                    </div>
                    <ul class="tips-list">
                        <li>Use good lighting conditions</li>
                        <li>Ensure face is clearly visible</li>
                        <li>Center face in the frame</li>
                        <li>Remove masks or face coverings</li>
                        <li>Use front-facing camera view</li>
                        <li>Avoid multiple faces in one image</li>
                    </ul>
                </div>
                """)
                
                # Credits in Sidebar
                gr.HTML("""
                <div class="sidebar-credits">
                    <div class="credits-title">Powered By</div>
                    <div class="tech-stack">
                        <div class="tech-item">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M21 16.25V7.75C21 7.02065 21 6.65597 20.8365 6.37109C20.6927 6.11652 20.4657 5.91738 20.1913 5.80866C19.8815 5.68251 19.4884 5.71681 18.7021 5.78541C18.5718 5.79792 18.5066 5.80417 18.4461 5.80598C18.1364 5.81601 17.8463 5.73293 17.6021 5.57148C17.5529 5.53698 17.5057 5.49601 17.4114 5.41407C17.0116 5.07221 16.8117 4.90127 16.5738 4.78227C16.3649 4.67776 16.1335 4.61389 15.8954 4.59523C15.6282 4.57433 15.3551 4.62455 14.8089 4.72498L12.3856 5.17661C11.9184 5.26333 11.6848 5.30669 11.4497 5.27137C11.2353 5.23942 11.0295 5.16599 10.8426 5.05475C10.6362 4.9313 10.4645 4.75958 10.1211 4.41615L9.80236 4.09743C9.48273 3.77781 9.32292 3.61799 9.14788 3.50453C8.99042 3.40262 8.81852 3.32646 8.63889 3.27839C8.43909 3.22574 8.22799 3.22574 7.8058 3.22574H7C6.44772 3.22574 6 3.67346 6 4.22574V19.7743C6 20.3265 6.44772 20.7743 7 20.7743H7.8058C8.22799 20.7743 8.43909 20.7743 8.63889 20.7216C8.81852 20.6736 8.99042 20.5974 9.14788 20.4955C9.32292 20.382 9.48273 20.2222 9.80236 19.9026L10.1211 19.5839C10.4645 19.2404 10.6362 19.0687 10.8426 18.9453C11.0295 18.834 11.2353 18.7606 11.4497 18.7286C11.6848 18.6933 11.9184 18.7367 12.3856 18.8234L14.8089 19.275C15.3551 19.3755 15.6282 19.4257 15.8954 19.4048C16.1335 19.3861 16.3649 19.3223 16.5738 19.2178C16.8117 19.0988 17.0116 18.9278 17.4114 18.586C17.5057 18.504 17.5529 18.4631 17.6021 18.4286C17.8463 18.2671 18.1364 18.184 18.4461 18.194C18.5066 18.1959 18.5718 18.2021 18.7021 18.2146C19.4884 18.2832 19.8815 18.3175 20.1913 18.1914C20.4657 18.0827 20.6927 17.8835 20.8365 17.629C21 17.344 21 16.9794 21 16.25Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M10 8.5H14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M10 12H14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M10 15.5H14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                            <span>TensorFlow</span>
                        </div>
                        <div class="tech-item">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M2 11V9C2 5.68629 4.68629 3 8 3H16C19.3137 3 22 5.68629 22 9V11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M12 9L16 13M16 13L12 17M16 13H2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M22 13L21 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M22 17L19 17" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M22 21L17 21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                            <span>Gradio</span>
                        </div>
                        <div class="tech-item">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M13 2.04932C13 2.04932 16 5.99994 16 11.9999" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M11 21.9506C11 21.9506 8 17.9999 8 11.9999" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M2.62964 15.5H12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M2.62964 8.5H21.3704" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path fill-rule="evenodd" clip-rule="evenodd" d="M21.8789 17.9174C22.3727 18.2211 22.3727 18.7788 21.8789 19.0825L19.9631 20.2412C19.4631 20.5489 18.8263 20.1952 18.8263 19.6587V17.3413C18.8263 16.8048 19.4631 16.4511 19.9631 16.7588L21.8789 17.9174Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <span>Deep Learning</span>
                        </div>
                    </div>
                </div>
                """)
            
            # Right Main Content Area
            with gr.Column(scale=3):
                # Top Section - Upload and Process Area
                with gr.Group(elem_classes="content-box main-upload-area"):
                    gr.HTML("""
                    <div class="section-header">
                        <div class="section-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M3 16.8V9.2C3 8.0799 3 7.51984 3.21799 7.09202C3.40973 6.71569 3.71569 6.40973 4.09202 6.21799C4.51984 6 5.0799 6 6.2 6H7.25464C7.37758 6 7.43905 6 7.49576 5.9935C7.79166 5.95961 8.05705 5.79559 8.21969 5.54609C8.25086 5.49827 8.27836 5.44328 8.33333 5.33333C8.44329 5.11342 8.49827 5.00346 8.56062 4.90782C8.8859 4.40882 9.41668 4.08078 10.0085 4.01299C10.1219 4 10.2448 4 10.4907 4H13.5093C13.7552 4 13.8781 4 13.9915 4.01299C14.5833 4.08078 15.1141 4.40882 15.4394 4.90782C15.5017 5.00345 15.5567 5.11345 15.6667 5.33333C15.7216 5.44329 15.7491 5.49827 15.7803 5.54609C15.943 5.79559 16.2083 5.95961 16.5042 5.9935C16.561 6 16.6224 6 16.7454 6H17.8C18.9201 6 19.4802 6 19.908 6.21799C20.2843 6.40973 20.5903 6.71569 20.782 7.09202C21 7.51984 21 8.0799 21 9.2V16.8C21 17.9201 21 18.4802 20.782 18.908C20.5903 19.2843 20.2843 19.5903 19.908 19.782C19.4802 20 18.9201 20 17.8 20H6.2C5.0799 20 4.51984 20 4.09202 19.782C3.71569 19.5903 3.40973 19.2843 3.21799 18.908C3 18.4802 3 17.9201 3 16.8Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M15 13L12.2165 10.2165V10.2165C12.0972 10.0972 11.9028 10.0972 11.7835 10.2165V10.2165L9 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M12 17V10.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="section-title">Upload Image for Analysis</div>
                    </div>
                    
                    <div class="upload-description">
                        Upload a clear photo of a face or take a picture with your webcam to detect the emotion.
                    </div>
                    """)
                    
                    # Upload Component
                    with gr.Row():
                        image_input = gr.Image(
                            label="",
                            type="pil",
                            sources=["upload", "webcam"],
                            height=280,
                            width=380,
                            show_label=False,
                            elem_classes="image-input-component"
                        )
                    
                    # Button Row
                    with gr.Row(elem_classes="action-buttons"):
                        predict_btn = gr.Button(
                            "Analyze Emotion", 
                            variant="primary",
                            elem_classes="analyze-button"
                        )
                        
                        clear_btn = gr.Button(
                            "Clear", 
                            elem_classes="clear-button"
                        )
                
                # Bottom Section - Results Area
                with gr.Group(elem_classes="content-box results-area"):
                    gr.HTML("""
                    <div class="section-header">
                        <div class="section-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M9 16C9.85038 16.6303 10.8846 17 12 17C13.1154 17 14.1496 16.6303 15 16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M9 11V10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                <path d="M15 11V10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <div class="section-title">Analysis Results</div>
                    </div>
                    """)
                    
                    # Results Display
                    result_display = gr.HTML(
                        value="""
                        <div class="initial-result-state">
                            <div class="result-placeholder">
                                <div class="result-placeholder-icon">
                                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M9 17C9.85038 17.6303 10.8846 18 12 18C13.1154 18 14.1496 17.6303 15 17" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                        <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" stroke-width="1.5"/>
                                        <path d="M9 15C9 15 10 14 12 14C14 14 15 15 15 15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                        <path d="M9 10H9.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                        <path d="M15 10H15.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                    </svg>
                                </div>
                                <div class="result-placeholder-title">Ready to Analyze</div>
                                <div class="result-placeholder-description">Upload or capture an image to detect facial emotion</div>
                            </div>
                        </div>
                        """,
                        elem_classes="result-display-component",
                        show_label=False
                    )
    
    # Tips Section
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                elem_classes="tips-section"
            )
     # Modern Minimal Footer
    gr.HTML("""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin: 40px 0 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(226, 232, 240, 0.8);
    ">
        <div style="color: #1e293b; font-size: 1.125rem; font-weight: 600; margin-bottom: 20px;">
            Powered by advanced deep learning technology
        </div>
        
        <div style="display: flex; justify-content: center; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 24px;">
            <!-- Tech Stack Pills -->
            <div style="
                display: flex; 
                align-items: center; 
                gap: 6px; 
                padding: 8px 16px; 
                background: rgba(99, 102, 241, 0.1); 
                border-radius: 8px; 
                border: 1px solid rgba(99, 102, 241, 0.2);
                transition: all 0.2s ease;
            ">
                <svg width="16" height="16" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M26 2H6C3.79086 2 2 3.79086 2 6V26C2 28.2091 3.79086 30 6 30H26C28.2091 30 30 28.2091 30 26V6C30 3.79086 28.2091 2 26 2Z" stroke="#6366f1" stroke-width="2"/>
                    <path d="M9 16H23M16 9V23" stroke="#6366f1" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <span style="color: #1e293b; font-size: 0.875rem; font-weight: 500;">TensorFlow</span>
            </div>
            
            <div style="
                display: flex; 
                align-items: center; 
                gap: 6px; 
                padding: 8px 16px; 
                background: rgba(139, 92, 246, 0.1); 
                border-radius: 8px; 
                border: 1px solid rgba(139, 92, 246, 0.2);
                transition: all 0.2s ease;
            ">
                <svg width="16" height="16" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 2C8.26801 2 2 8.26801 2 16C2 23.732 8.26801 30 16 30C23.732 30 30 23.732 30 16C30 8.26801 23.732 2 16 2Z" stroke="#8b5cf6" stroke-width="2"/>
                    <path d="M11 16H21M16 11V21" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <span style="color: #1e293b; font-size: 0.875rem; font-weight: 500;">Gradio</span>
            </div>
            
            <div style="
                display: flex; 
                align-items: center; 
                gap: 6px; 
                padding: 8px 16px; 
                background: rgba(16, 185, 129, 0.1); 
                border-radius: 8px; 
                border: 1px solid rgba(16, 185, 129, 0.2);
                transition: all 0.2s ease;
            ">
                <svg width="16" height="16" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M6 16H26M16 6V26" stroke="#10b981" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <span style="color: #1e293b; font-size: 0.875rem; font-weight: 500;">Deep Learning</span>
            </div>
        </div>
        
        <div style="border-top: 1px solid #e2e8f0; padding-top: 20px;">
            <p style="color: #64748b; margin: 0; font-size: 0.875rem; line-height: 1.6;">
                This application analyzes facial expressions to identify emotions.<br>
                Based on a convolutional neural network trained on facial images.
            </p>
        </div>
    </div>
    
    <script>
        // Add hover effects to tech stack pills
        document.querySelectorAll('[style*="display: flex; align-items: center; gap: 6px;"]').forEach(pill => {
            pill.addEventListener('mouseenter', () => {
                pill.style.transform = 'translateY(-2px)';
                pill.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
            });
            
            pill.addEventListener('mouseleave', () => {
                pill.style.transform = 'translateY(0)';
                pill.style.boxShadow = 'none';
            });
        });
    </script>
    """)
    
    # Event Handlers - Single output
    predict_btn.click(
        fn=predict_interface,
        inputs=image_input,
        outputs=result_display,  # Only one output
        show_progress=True
    )

# Modern Clear Button
    clear_btn.click(
        fn=lambda: (None, """
            <div style="
                text-align: center;
                padding: 48px 24px;
                background: #1f2937;
                border-radius: 12px;
                color: #f9fafb;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border: 1px solid rgba(75, 85, 99, 0.4);
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            ">
                <div style="
                    width: 80px;
                    height: 80px;
                    margin: 0 auto 24px auto;
                    background: #374151;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M9 16C9.85038 16.6303 10.8846 17 12 17C13.1154 17 14.1496 16.6303 15 16" stroke="#9ca3af" stroke-width="1.5" stroke-linecap="round"/>
                        <path d="M12 13V13.01" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"/>
                        <path d="M15 8H15.01" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"/>
                        <path d="M9 8H9.01" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"/>
                        <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#9ca3af" stroke-width="1.5"/>
                    </svg>
                </div>
                <h3 style="margin: 0 0 8px 0; font-size: 20px; font-weight: 600; color: #f9fafb;">Ready to Analyze</h3>
                <p style="margin: 0; font-size: 16px; color: #9ca3af;">Upload an image or take a photo to detect emotions</p>
            </div>
            """),
        outputs=[image_input, result_display]
    )

if __name__ == "__main__":
    demo.launch(
        debug=True,           # Enable debug mode
        show_error=True,      # Show detailed errors  
        inbrowser=True,       # Auto-open browser
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,     # Fixed port
        share=True            # Create shareable link
    )