"""
Setup script for downloading and preparing Mistral 7B LLM
Run this once before using the local LLM option in PodcastGen
"""

import os
import torch

def download_mistral_model():
    """Download and cache Mistral 7B model with 4-bit quantization"""
    print("📥 Downloading Mistral 7B model (this may take 2-3 minutes on first run)...")
    print("   Model: mistralai/Mistral-7B-Instruct-v0.3")
    print("   License: Apache 2.0 (commercial-friendly)")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # 4-bit quantization config for T4 GPU (uses ~3.5GB VRAM)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("\n🔧 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            trust_remote_code=True
        )
        
        print("🔧 Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("\n✅ Mistral 7B model downloaded and ready!")
        print(f"   GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nMake sure you have installed:")
        print("  pip install transformers accelerate bitsandbytes")
        return None, None


def test_generation(model, tokenizer):
    """Test the model with a simple generation"""
    if model is None:
        return
        
    print("\n📝 Testing generation...")
    
    messages = [
        {"role": "user", "content": "Write a short 2-line podcast script about AI in Hindi with English technical words kept in English."}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Test Output ---")
    print(response)
    print("--- End ---\n")


if __name__ == "__main__":
    print("="*60)
    print("   Mistral 7B Setup for PodcastGen")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️ No GPU found! Model will run on CPU (very slow)")
    
    # Download model
    model, tokenizer = download_mistral_model()
    
    # Test generation
    if model:
        test_generation(model, tokenizer)
        print("\n🎉 Setup complete! You can now use 'Mistral Local' in PodcastGen.")
