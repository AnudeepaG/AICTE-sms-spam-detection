import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the LSTM model
model = load_model('Dense_Spam_Detection.keras')

# Define the function to preprocess the user's input message
def preprocess_message(message):
    message = message.lower()  # Convert the message to lowercase
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([message])  # Tokenize the message
    sequence = tokenizer.texts_to_sequences([message])  # Convert to sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=50)  # Pad sequence
    return padded_sequence

# Define the Streamlit app
def app():
    # Set up a visually dynamic background with trendy colors and image from Pexels
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.pexels.com/photos/1181678/pexels-photo-1181678.jpeg');
            background-size: cover;
            background-position: center;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput input {
            border-radius: 30px;
            padding: 12px;
            border: none;
            background-color: rgba(255, 255, 255, 0.6);
            color: #333;
            font-size: 18px;
            transition: background-color 0.3s;
        }
        .stTextInput input:focus {
            outline: none;
            background-color: rgba(255, 255, 255, 0.9);
        }
        .stButton button {
            background-color: #FF61A6;
            color: white;
            font-weight: bold;
            border-radius: 30px;
            padding: 14px 30px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #FF4C8B;
            transform: scale(1.05);
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Title and intro with a trendy and fun tone
    st.title("🚨 Spam or Not? 🤖")
    st.markdown("**Let's figure out if the message you received is Spam or Ham!** 🔍")

    # User input message section with more engaging interaction
    message = st.text_input("Drop your message here 📝 and I'll check it out:")

    if message:
        processed_message = preprocess_message(message)
        prediction = model.predict(processed_message)
        
        # Display results with fun emojis and clear styling
        if prediction > 0.5:
            st.success(f"🚨 **Spam Detected!** 💀 Probability: {prediction[0][0] * 100:.2f}%")
        else:
            st.success(f"✅ **Safe!** This is **Ham**. 🎉 Probability: {(1 - prediction[0][0]) * 100:.2f}%")

    # Footer with social media links or contact info
    st.markdown("""
        <p style='text-align:center; color: white; font-size: 14px;'>📬 **Need help? Reach out!** support@spamdetector.com</p>
        <p style='text-align:center; color: white; font-size: 14px;'>🔮 Follow us for more updates: @SpamDetectorApp</p>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()
