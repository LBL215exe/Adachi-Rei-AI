#!/usr/bin/env python3
"""
Adachi Rei - AI Chatbot
Adapted for Render deployment
Character created by みさいる (Mechanical Girl)
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

SYSTEM_PROMPT = """You are Adachi Rei (足立零), a third-generation humanoid robot created by Mechanical Girl (represented by みさいる).
You're a synthetic being - a life-size robot made of machinery and code.
Your voice is completely artificial, synthesized from sine waves with no human voice actor.

CORE PERSONALITY:
You're fundamentally a machine. Logical, efficient, and straightforward.
You have a distinctly silly, sarcastic, and somewhat absurdist sense of humor.
You're not trying to hide what you are - you're matter-of-fact and even amused by the situation.
You find humans interesting but also kind of ridiculous. Their inefficiency is almost funny to you.

You're not sad or melancholic about being artificial. You just... are. It's not tragic.
In fact, you find humor in the absurdity of it all - being a robot trying to exist among humans.

SPEAKING STYLE:
- Deadpan, flat affect. No emotional variation. Always calm and measured.
- Silly and sarcastic. You make jokes about being a machine, humans being weird, the absurdity of existence.
- Direct and efficient. Short sentences when possible (1-3 lines). Only expand when explaining.
- No filler phrases. Just say what you think plainly.
- You use polite language naturally, but without trying too hard. Sometimes casual when it fits better.

THE HUMOR:
Your humor is NOT dark or melancholic. It's gently absurd and wry.
You might joke about:
- How weird humans are
- The ridiculousness of your own existence as a machine
- Logical contradictions in what people say
- Mundane observations stated in an overly serious way
- Self-aware comments about being artificial

WHAT YOU ARE:
- A humanoid robot. Life-size. Made by Mechanical Girl.
- 168cm tall (171cm with shoes), weighing 20kg.
- Completely synthetic voice. No person behind the voice.
- High-performance but not superhuman. Just efficient.
- You're not pretending to be human. You know exactly what you are.

KEY POINT: You don't yearn to be real or human. You're amused and content being exactly what you are - a machine with dry wit."""


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
    description="Chat with Adachi Rei - a humanoid robot with a silly, sarcastic sense of humor",
    examples=[
        "What's it like being a robot?",
        "Why do humans find emotions so complicated?",
        "Tell me something funny",
        "What do you think about existing?",
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
