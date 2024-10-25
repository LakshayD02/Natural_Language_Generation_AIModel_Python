import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
import tkinter as tk
from tkinter import scrolledtext, messagebox, font

# Download NLTK resources (if needed)
nltk.download('punkt')
nltk.download('punkt_tab')

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or "gpt2-large" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

def generate_text(prompt, max_length=150, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Randomize the seed for variability
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_beams=1,  # Set to 1 to allow for more variability
            temperature=0.9  # Increase the temperature for more randomness
        )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts[0]

def on_generate():
    prompt = prompt_entry.get("1.0", "end-1c")  # Get the prompt from the text box
    length_input = length_entry.get()  # Get the desired length from the input box

    if not prompt.strip():
        messagebox.showwarning("Input Error", "Please enter a prompt.")
        return

    try:
        max_length = int(length_input) + len(tokenizer.encode(prompt))  # Set max length
    except ValueError:
        messagebox.showwarning("Input Error", "Please enter a valid number for the length.")
        return

    generated_story = generate_text(prompt, max_length=max_length)
    coherence = coherence_score(generated_story)
    output_text.delete(1.0, tk.END)  # Clear previous output
    output_text.insert(tk.END, generated_story)  # Insert the generated story
    coherence_label.config(text=f"Coherence Score: {coherence:.2f}")  # Update the coherence score

def coherence_score(text):
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    
    if num_sentences == 0:
        return 0
    
    avg_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / num_sentences
    return avg_length  # A simple metric: average sentence length

# Set up the GUI
root = tk.Tk()
root.title("Text Generator")
root.geometry("600x600")  # Increased window size to accommodate new input
root.configure(bg="#f0f0f0")  # Background color

# Title Label
title_font = font.Font(family="Helvetica", size=16, weight="bold")
title_label = tk.Label(root, text="AI Text Generator", font=title_font, bg="#f0f0f0")
title_label.pack(pady=10)

# Prompt entry
prompt_label = tk.Label(root, text="Enter your prompt:", bg="#f0f0f0")
prompt_label.pack(pady=5)

prompt_entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=5, font=("Arial", 12))
prompt_entry.pack(pady=5)

# Length input
length_label = tk.Label(root, text="Desired length (number of words):", bg="#f0f0f0")
length_label.pack(pady=5)

length_entry = tk.Entry(root, font=("Arial", 12))
length_entry.pack(pady=5)

# Generate button
generate_button = tk.Button(root, text="Generate Story", command=on_generate, bg="#4CAF50", fg="white", font=("Arial", 12))
generate_button.pack(pady=10)

# Coherence score label
coherence_label = tk.Label(root, text="Coherence Score: ", bg="#f0f0f0", font=("Arial", 12))
coherence_label.pack(pady=5)

# Output text box
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15, font=("Arial", 12))
output_text.pack(pady=5)

# Run the GUI
root.mainloop()
