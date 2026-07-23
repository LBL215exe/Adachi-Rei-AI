#!/usr/bin/env python3
"""
Gradio app for Adachi-Rei-AI
"""
import gradio as gr
import os

# Your AI logic here
def chat(message, history):
    """Main chat function"""
    # Replace with your actual AI logic
    response = f"Echo: {message}"
    return response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Adachi-Rei-AI Chatbot")
    
    chatbot = gr.ChatInterface(
        chat,
        examples=["Hello!", "How are you?"],
        title="Chat with Adachi-Rei",
        description="Talk to the Adachi-Rei AI assistant"
    )

if __name__ == "__main__":
    # Get port from environment or use 7860
    port = int(os.getenv("PORT", 7860))
    
    # Launch with these settings for Render
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
