import torch
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline

def load_models():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    return model, processor, pipe

def generate_image_from_text(text, model, processor, pipe, save_path="output.png"):
    if not text:
        st.warning("Please enter a text description to generate an image.")
        return None
    
    inputs = processor(text=[text], return_tensors="pt")
    st.write("Generating image, please wait...")
    
    try:
        image = pipe(text).images[0]
        image.save(save_path)
        st.success("Image generated successfully!")
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None
    
    # Display the generated image
    st.image(save_path, caption="Generated Image")
    return save_path

if __name__ == "__main__":
    st.title("Text-to-Image Generator")
    st.write("Enter a text description and generate an AI-generated image!")
    
    with st.spinner("Loading models..."):
        try:
            model, processor, pipe = load_models()
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    input_text = st.text_input("Enter a text description:")
    
    if st.button("Generate Image"):
        save_path = generate_image_from_text(input_text, model, processor, pipe)
