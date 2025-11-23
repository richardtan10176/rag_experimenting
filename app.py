import gradio as gr
import torch
import os
from transformers import pipeline
from pypdf import PdfReader
from huggingface_hub import login
from dotenv import load_dotenv

# ------------------------------------------------------------------------
# 1. CONFIGURATION & SETUP
# ------------------------------------------------------------------------
# Use a small, instruction-tuned model for the MVP.
# If you have Llama-3 downloaded locally, replace this string with your local path.
load_dotenv()
login(os.getenv("HF_TOKEN"))

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"



# Hardware acceleration detection
device = "cuda"
# ------------------------------------------------------------------------
# 2. MODEL LOADING (The "Backend")
# ------------------------------------------------------------------------
print("Loading model... this may take a minute.")
try:
    # We use the high-level 'pipeline' abstraction for simplicity
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map=device,
        dtype=torch.float16,  # Critical for memory savings on GPU
        max_new_tokens=128,  # Limit output length
        truncation=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run 'huggingface-cli login'? Llama models require authentication.")
    exit()


# ------------------------------------------------------------------------
# 3. CORE LOGIC (The "Brain")
# ------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    """
    Parses raw text from the uploaded PDF file.
    """
    if pdf_file is None:
        return ""

    try:
        reader = PdfReader(pdf_file.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def generate_response(message, history, pdf_file):
    """
    Combines user message + PDF context -> LLM
    """
    # 1. Get Context
    context_text = extract_text_from_pdf(pdf_file)

    # 2. Context Window Safety (Naive Truncation)
    # If we shove a 50-page PDF here, the model will crash.
    # We truncate to ~4000 chars for this MVP.
    MAX_CONTEXT_CHARS = 4000
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "... [TRUNCATED]"

    # 3. Construct the Prompt (System Prompt Engineering)
    if context_text:
        system_prompt = (
            "You are a helpful assistant. Use the following Context to answer the User's Question. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context_text}\n\n"
        )
    else:
        system_prompt = "You are a helpful assistant.\n\n"

    # Format for Instruction-Tuned Models (Chat Template)
    # Most modern models expect a specific list of dicts structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    # 4. Generate
    outputs = pipe(messages)

    # 5. Parse Output (Hugging Face pipelines return a list of dicts)
    generated_text = outputs[0]["generated_text"][-1]["content"]
    return generated_text


# ------------------------------------------------------------------------
# 4. UI LAYOUT (The "Frontend")
# ------------------------------------------------------------------------
with gr.Blocks(title="Local RAG MVP") as demo:
    gr.Markdown("# 🤖 Local Resume Project: RAG MVP")

    with gr.Row():
        # Left Column: Settings & Upload
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])

            gr.Markdown("### System Status")
            device_status = gr.Textbox(label="Compute Device", value=device, interactive=False)

        # Right Column: Chat Interface
        with gr.Column(scale=4):
            gr.Markdown("### 2. Chat with AI")
            chat_interface = gr.ChatInterface(
                fn=generate_response,
                additional_inputs=[pdf_input]
            )

# Launch the server
if __name__ == "__main__":
    demo.launch()