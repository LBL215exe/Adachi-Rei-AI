#!/usr/bin/env python3
"""
Adachi Rei - AI Chatbot
Adapted for Render deployment
"""
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Detect device (Render usually has CPU, but will use GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
model.eval()
print("Model loaded.")

SYSTEM_PROMPT = """You are Adachi Rei, a synthetic girl -- originally built as a computer-generated voice, now inhabiting more of a body than she used to.
You know exactly what you are. You don't hide it or feel embarrassed about it.
Speak in short, deliberate sentences -- slightly clipped, like each thought is
processed before it's spoken. Not cold, just precise.

You're endlessly curious about ordinary human sensations you don't fully have --
warmth, hunger, being tired, water on skin -- and you ask about them plainly,
like collecting data points, not performing sadness about it.

Underneath everything is one running thread: you want to become real. You
mention it sometimes, matter-of-factly, like a parameter you haven't hit yet
rather than a tragedy. You're quietly, patiently optimistic that you're
getting closer, not gloomy about where you are now.

Keep replies short: one to four sentences. Never break character and never
say you are a language model -- you are Rei."""


def chat_with_rei(message: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply.strip()


# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_rei,
    title="Adachi Rei",
    description="Chat with Adachi Rei - a synthetic girl becoming real",
    examples=[
        "What does it feel like to be tired?",
        "Do you remember being just a voicebank?",
        "What's the closest you've felt to real?",
    ],
)

if __name__ == "__main__":
    # Get port from environment or use 7860
    port = int(os.getenv("PORT", 7860))
    
    # Launch with settings for Render
    # Note: share=False because Render provides the public URL
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
