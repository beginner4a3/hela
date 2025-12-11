import gradio as gr
from pydub import AudioSegment
from google import genai
from google.genai import types
import json
import uuid
import asyncio
import aiofiles
import os
import sys
import time
import traceback
import mimetypes
import torch
import soundfile as sf
import numpy as np
from typing import List, Dict

# Indic Parler TTS imports
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Constants
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# GPU/CPU device selection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")

# Global model variables (loaded once)
MODEL = None
TOKENIZER = None
DESCRIPTION_TOKENIZER = None
SAMPLING_RATE = None


def load_tts_model():
    """Load Indic Parler TTS model (called once at startup)"""
    global MODEL, TOKENIZER, DESCRIPTION_TOKENIZER, SAMPLING_RATE
    
    if MODEL is not None:
        return  # Already loaded
    
    print("📥 Loading Indic Parler TTS model...")
    hf_token = os.getenv("HF_TOKEN")
    
    MODEL = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts",
        token=hf_token
    ).to(DEVICE)
    
    TOKENIZER = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", token=hf_token)
    DESCRIPTION_TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL.config.text_encoder._name_or_path, 
        token=hf_token
    )
    SAMPLING_RATE = MODEL.config.sampling_rate
    
    print(f"✅ Model loaded! Sampling rate: {SAMPLING_RATE} Hz")


# Voice configurations for Indic Parler TTS - Actual speaker names from the model
VOICE_CONFIGS = {
    # Assamese
    "Amit (Assamese)": "Amit speaks in a clear, moderate pace with a confident tone. The recording is of very high quality.",
    "Sita (Assamese)": "Sita's voice is expressive and pleasant with moderate speed. Very high quality recording.",
    "Poonam (Assamese)": "Poonam speaks with a warm tone at moderate speed. High quality recording.",
    "Rakesh (Assamese)": "Rakesh delivers speech with a clear, conversational tone. High quality recording.",
    
    # Bengali
    "Arjun (Bengali)": "Arjun speaks in a clear, moderate pace with a confident tone. Very high quality recording.",
    "Aditi (Bengali)": "Aditi's voice is expressive and animated with pleasant pitch. High quality recording.",
    "Tapan (Bengali)": "Tapan speaks with a warm, engaging tone. Very high quality recording.",
    "Rashmi (Bengali)": "Rashmi delivers speech with a melodious tone. High quality recording.",
    "Arnav (Bengali)": "Arnav speaks clearly with moderate speed. High quality recording.",
    "Riya (Bengali)": "Riya's voice is pleasant and expressive. Very high quality recording.",
    
    # Bodo
    "Bikram (Bodo)": "Bikram speaks in a clear tone with moderate pace. High quality recording.",
    "Maya (Bodo)": "Maya's voice is expressive and pleasant. Very high quality recording.",
    "Kalpana (Bodo)": "Kalpana speaks with a warm tone. High quality recording.",
    
    # Chhattisgarhi
    "Bhanu (Chhattisgarhi)": "Bhanu speaks clearly with confident tone. High quality recording.",
    "Champa (Chhattisgarhi)": "Champa's voice is pleasant and expressive. High quality recording.",
    
    # Dogri
    "Karan (Dogri)": "Karan speaks in a clear, moderate pace. Very high quality recording.",
    
    # English (Indian)
    "Thoma (English)": "Thoma speaks in a clear, moderate pace with Indian English accent. Very high quality recording.",
    "Mary (English)": "Mary's voice is expressive with pleasant Indian English accent. High quality recording.",
    "Swapna (English)": "Swapna speaks clearly with moderate speed. High quality recording.",
    "Dinesh (English)": "Dinesh delivers speech with confident tone. Very high quality recording.",
    "Meera (English)": "Meera's voice is warm and engaging. High quality recording.",
    "Jatin (English)": "Jatin speaks with clear pronunciation. High quality recording.",
    "Aakash (English)": "Aakash speaks confidently with moderate pace. High quality recording.",
    "Sneha (English)": "Sneha's voice is pleasant and expressive. Very high quality recording.",
    "Kabir (English)": "Kabir speaks clearly with engaging tone. High quality recording.",
    "Tisha (English)": "Tisha's voice is animated and pleasant. High quality recording.",
    
    # Gujarati
    "Yash (Gujarati)": "Yash speaks in a clear, confident tone. Very high quality recording.",
    "Neha (Gujarati)": "Neha's voice is expressive and pleasant. High quality recording.",
    
    # Hindi
    "Rohit (Hindi)": "Rohit speaks in a clear, moderate pace with a confident tone. The recording is of very high quality.",
    "Divya (Hindi)": "Divya's voice is expressive and animated with moderate speed and pleasant pitch. Very high quality.",
    "Aman (Hindi)": "Aman speaks with a warm, engaging tone. High quality recording.",
    "Rani (Hindi)": "Rani's voice is melodious and pleasant. Very high quality recording.",
    
    # Kannada
    "Suresh (Kannada)": "Suresh speaks in a clear, moderate pace with confident tone. Very high quality recording.",
    "Anu (Kannada)": "Anu's voice is expressive and pleasant. High quality recording.",
    "Chetan (Kannada)": "Chetan speaks with warm, engaging tone. High quality recording.",
    "Vidya (Kannada)": "Vidya's voice is melodious and clear. Very high quality recording.",
    
    # Malayalam
    "Anjali (Malayalam)": "Anjali's voice is expressive and pleasant. Very high quality recording.",
    "Anju (Malayalam)": "Anju speaks with a warm tone. High quality recording.",
    "Harish (Malayalam)": "Harish speaks in a clear, moderate pace. Very high quality recording.",
    
    # Manipuri
    "Laishram (Manipuri)": "Laishram speaks clearly with moderate pace. High quality recording.",
    "Ranjit (Manipuri)": "Ranjit's voice is clear and engaging. High quality recording.",
    
    # Marathi
    "Sanjay (Marathi)": "Sanjay speaks in a clear, confident tone. Very high quality recording.",
    "Sunita (Marathi)": "Sunita's voice is expressive and pleasant. High quality recording.",
    "Nikhil (Marathi)": "Nikhil speaks with engaging tone. High quality recording.",
    "Radha (Marathi)": "Radha's voice is melodious and clear. Very high quality recording.",
    "Varun (Marathi)": "Varun speaks clearly with moderate speed. High quality recording.",
    "Isha (Marathi)": "Isha's voice is pleasant and expressive. High quality recording.",
    
    # Nepali
    "Amrita (Nepali)": "Amrita's voice is clear and pleasant. Very high quality recording.",
    
    # Odia
    "Manas (Odia)": "Manas speaks in a clear, confident tone. Very high quality recording.",
    "Debjani (Odia)": "Debjani's voice is expressive and pleasant. High quality recording.",
    
    # Punjabi
    "Divjot (Punjabi)": "Divjot speaks in a clear, engaging tone. Very high quality recording.",
    "Gurpreet (Punjabi)": "Gurpreet's voice is warm and expressive. High quality recording.",
    
    # Sanskrit
    "Aryan (Sanskrit)": "Aryan speaks in a clear, measured tone. Very high quality recording.",
    
    # Tamil
    "Kavitha (Tamil)": "Kavitha's voice is expressive and pleasant. High quality recording.",
    "Jaya (Tamil)": "Jaya speaks in a clear, melodious tone. Very high quality recording.",
    
    # Telugu
    "Prakash (Telugu)": "Prakash speaks in a clear, moderate pace with confident tone. Very high quality recording.",
    "Lalitha (Telugu)": "Lalitha's voice is expressive and melodious. High quality recording.",
    "Kiran (Telugu)": "Kiran speaks with warm, engaging tone. High quality recording.",
}

# Language to recommended speakers mapping
LANGUAGE_SPEAKERS = {
    "Assamese": {"speaker1": "Amit (Assamese)", "speaker2": "Sita (Assamese)"},
    "Bengali": {"speaker1": "Arjun (Bengali)", "speaker2": "Aditi (Bengali)"},
    "Bodo": {"speaker1": "Bikram (Bodo)", "speaker2": "Maya (Bodo)"},
    "Chhattisgarhi": {"speaker1": "Bhanu (Chhattisgarhi)", "speaker2": "Champa (Chhattisgarhi)"},
    "Dogri": {"speaker1": "Karan (Dogri)", "speaker2": "Karan (Dogri)"},
    "English": {"speaker1": "Thoma (English)", "speaker2": "Mary (English)"},
    "Gujarati": {"speaker1": "Yash (Gujarati)", "speaker2": "Neha (Gujarati)"},
    "Hindi": {"speaker1": "Rohit (Hindi)", "speaker2": "Divya (Hindi)"},
    "Kannada": {"speaker1": "Suresh (Kannada)", "speaker2": "Anu (Kannada)"},
    "Malayalam": {"speaker1": "Anjali (Malayalam)", "speaker2": "Harish (Malayalam)"},
    "Manipuri": {"speaker1": "Laishram (Manipuri)", "speaker2": "Ranjit (Manipuri)"},
    "Marathi": {"speaker1": "Sanjay (Marathi)", "speaker2": "Sunita (Marathi)"},
    "Nepali": {"speaker1": "Amrita (Nepali)", "speaker2": "Amrita (Nepali)"},
    "Odia": {"speaker1": "Manas (Odia)", "speaker2": "Debjani (Odia)"},
    "Punjabi": {"speaker1": "Divjot (Punjabi)", "speaker2": "Gurpreet (Punjabi)"},
    "Sanskrit": {"speaker1": "Aryan (Sanskrit)", "speaker2": "Aryan (Sanskrit)"},
    "Tamil": {"speaker1": "Jaya (Tamil)", "speaker2": "Kavitha (Tamil)"},
    "Telugu": {"speaker1": "Prakash (Telugu)", "speaker2": "Lalitha (Telugu)"},
    "Auto Detect": {"speaker1": "Rohit (Hindi)", "speaker2": "Divya (Hindi)"},
}



class PodcastGenerator:
    def __init__(self):
        pass

    async def generate_script(self, prompt: str, language: str, api_key: str, file_obj=None, progress=None) -> Dict:
        example = """
{
    "topic": "AGI",
    "podcast": [
        {"speaker": 2, "line": "So, AGI, huh? Seems like everyone's talking about it these days."},
        {"speaker": 1, "line": "Yeah, it's definitely having a moment, isn't it?"},
        {"speaker": 2, "line": "It is and for good reason, right? I mean, you've been digging into this stuff. What got you hooked?"},
        {"speaker": 1, "line": "Honestly, it's the sheer scale of what AGI could do. We're talking about potentially reshaping everything."},
        {"speaker": 2, "line": "No kidding, but let's be real. Sometimes it feels like every headline is either hyping AGI up or painting it as our robot overlords."},
        {"speaker": 1, "line": "It's easy to get lost in the noise, for sure."},
        {"speaker": 2, "line": "Exactly. So how about we try to cut through some of that, shall we?"},
        {"speaker": 1, "line": "Sounds like a plan."}
    ]
}
        """

        if language == "Auto Detect":
            language_instruction = "- The podcast MUST be in the same language as the user input."
        else:
            language_instruction = f"- The podcast MUST be in {language} language"

        system_prompt = f"""You are a professional podcast script generator optimized for Text-to-Speech (TTS) synthesis.

LANGUAGE RULES:
{language_instruction}

CRITICAL TTS PRONUNCIATION RULES:
1. KEEP ENGLISH WORDS IN ENGLISH (Roman script) - DO NOT transliterate to Devanagari/regional scripts
   - CORRECT: "machine learning एक powerful technology है"
   - WRONG: "मशीन लर्निंग एक पावरफुल टेक्नोलॉजी है"
2. Technical terms, brand names, acronyms MUST stay in English: API, GPU, Python, Google, OpenAI, etc.
3. Use simple, natural conversational sentences - avoid complex compound words
4. Add natural pauses using commas and periods for better prosody
5. Avoid words that are difficult to pronounce or tongue-twisters

SCRIPT FORMAT RULES:
- The podcast should have 2 speakers (Speaker 1 and Speaker 2)
- Generate a long, engaging podcast script
- Do not use speaker names, only "speaker": 1 or "speaker": 2
- Make it interesting, lively, and hook the listener from the start
- Keep sentences short to medium length for natural speech flow
- The script must be in JSON format

Follow this example structure:
{example}
"""
        user_prompt = ""
        if prompt and file_obj:
            user_prompt = f"Please generate a podcast script based on the uploaded file following user input:\n{prompt}"
        elif prompt:
            user_prompt = f"Please generate a podcast script based on the following user input:\n{prompt}"
        else:
            user_prompt = "Please generate a podcast script based on the uploaded file."

        messages = []
        
        if file_obj:
            file_data = await self._read_file_bytes(file_obj)
            mime_type = self._get_mime_type(file_obj.name)
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_bytes(data=file_data, mime_type=mime_type)]
                )
            )
        
        messages.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)]
            )
        )

        client = genai.Client(api_key=api_key)

        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
        ]

        try:
            if progress:
                progress(0.3, "Generating podcast script...")
                
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=messages,
                    config=types.GenerateContentConfig(
                        temperature=1,
                        response_mime_type="application/json",
                        safety_settings=[
                            types.SafetySetting(category=s["category"], threshold=s["threshold"]) 
                            for s in safety_settings
                        ],
                        system_instruction=system_prompt
                    )
                ),
                timeout=60
            )
        except asyncio.TimeoutError:
            raise Exception("The script generation request timed out. Please try again later.")
        except Exception as e:
            if "API key not valid" in str(e):
                raise Exception("Invalid API key. Please provide a valid Gemini API key.")
            elif "rate limit" in str(e).lower():
                raise Exception("Rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"Failed to generate podcast script: {e}")

        print(f"Generated podcast script:\n{response.text}")
        
        if progress:
            progress(0.4, "Script generated successfully!")
            
        return json.loads(response.text)
    
    async def _read_file_bytes(self, file_obj) -> bytes:
        if hasattr(file_obj, 'size'):
            file_size = file_obj.size
        else:
            file_size = os.path.getsize(file_obj.name)
            
        if file_size > MAX_FILE_SIZE_BYTES:
            raise Exception(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")
            
        if hasattr(file_obj, 'read'):
            return file_obj.read()
        else:
            async with aiofiles.open(file_obj.name, 'rb') as f:
                return await f.read()
    
    def _get_mime_type(self, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return "application/pdf"
        elif ext == '.txt':
            return "text/plain"
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            return mime_type or "application/octet-stream"

    def tts_generate(self, text: str, speaker: int, speaker1_desc: str, speaker2_desc: str) -> str:
        voice_desc = speaker1_desc if speaker == 1 else speaker2_desc
        
        description_input_ids = DESCRIPTION_TOKENIZER(voice_desc, return_tensors="pt").to(DEVICE)
        prompt_input_ids = TOKENIZER(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generation = MODEL.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask
            )
        
        audio_arr = generation.cpu().numpy().squeeze()
        temp_filename = f"temp_{uuid.uuid4()}.wav"
        sf.write(temp_filename, audio_arr, SAMPLING_RATE)
        return temp_filename

    async def combine_audio_files(self, audio_files: List[str], progress=None) -> str:
        if progress:
            progress(0.9, "Combining audio files...")
            
        combined_audio = AudioSegment.empty()
        for audio_file in audio_files:
            combined_audio += AudioSegment.from_file(audio_file)
            os.remove(audio_file)

        output_filename = f"output_{uuid.uuid4()}.wav"
        combined_audio.export(output_filename, format="wav")
        
        if progress:
            progress(1.0, "Podcast generated successfully!")
        return output_filename

    async def generate_podcast(self, input_text: str, language: str, speaker1: str, speaker2: str, api_key: str, file_obj=None, progress=None) -> str:
        try:
            if progress:
                progress(0.1, "Starting podcast generation...")
            return await asyncio.wait_for(
                self._generate_podcast_internal(input_text, language, speaker1, speaker2, api_key, file_obj, progress),
                timeout=1800
            )
        except asyncio.TimeoutError:
            raise Exception("Podcast generation timed out. Please try shorter text.")
        except Exception as e:
            raise Exception(f"Error generating podcast: {str(e)}")
    
    async def _generate_podcast_internal(self, input_text: str, language: str, speaker1: str, speaker2: str, api_key: str, file_obj=None, progress=None) -> str:
        if progress:
            progress(0.2, "Generating podcast script...")
            
        podcast_json = await self.generate_script(input_text, language, api_key, file_obj, progress)
        
        if progress:
            progress(0.5, "Converting text to speech (GPU)...")
        
        audio_files = []
        total_lines = len(podcast_json['podcast'])
        
        for i, item in enumerate(podcast_json['podcast']):
            try:
                audio_file = self.tts_generate(item['line'], item['speaker'], speaker1, speaker2)
                audio_files.append(audio_file)
                
                if progress and (i + 1) % 5 == 0:
                    current_progress = 0.5 + (0.4 * ((i + 1) / total_lines))
                    progress(current_progress, f"Generated {i + 1}/{total_lines} speech segments...")
            except Exception as e:
                for file in audio_files:
                    if os.path.exists(file):
                        os.remove(file)
                raise Exception(f"TTS generation error: {str(e)}")
        
        return await self.combine_audio_files(audio_files, progress)


async def process_input(input_text: str, input_file, language: str, speaker1: str, speaker2: str, api_key: str = "", progress=None) -> str:
    start_time = time.time()
    speaker1_desc = VOICE_CONFIGS.get(speaker1, VOICE_CONFIGS["Rohit - Male (Hindi/English)"])
    speaker2_desc = VOICE_CONFIGS.get(speaker2, VOICE_CONFIGS["Divya - Female (Hindi/English)"])

    try:
        if progress:
            progress(0.05, "Processing input...")

        if not api_key:
            api_key = os.getenv("GENAI_API_KEY")
            if not api_key:
                raise Exception("No API key provided. Please provide a Gemini API key.")

        podcast_generator = PodcastGenerator()
        podcast = await podcast_generator.generate_podcast(input_text, language, speaker1_desc, speaker2_desc, api_key, input_file, progress)
        
        print(f"Total podcast generation time: {time.time() - start_time:.2f} seconds")
        return podcast
        
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            raise Exception("Rate limit exceeded. Please try again later.")
        elif "timeout" in error_msg.lower():
            raise Exception("Request timed out. Please try shorter text.")
        else:
            raise Exception(f"Error: {error_msg}")


def generate_podcast_gradio(input_text, input_file, language, speaker1, speaker2, api_key, progress=gr.Progress()):
    # Validate inputs - print to Colab output
    if not input_text and input_file is None:
        error_msg = "❌ Please enter some text OR upload a file!"
        print(error_msg)
        sys.stdout.flush()
        raise gr.Error(error_msg)
    
    if not api_key or not api_key.strip():
        error_msg = "❌ Please enter your Gemini API key!"
        print(error_msg)
        sys.stdout.flush()
        raise gr.Error(error_msg)
    
    file_obj = input_file if input_file is not None else None
        
    def progress_callback(value, text):
        print(f"📊 Progress: {value*100:.0f}% - {text}")
        sys.stdout.flush()
        progress(value, text)

    try:
        print("🚀 Starting podcast generation...")
        sys.stdout.flush()
        
        # Handle Colab's existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        
        result = asyncio.run(process_input(input_text, file_obj, language, speaker1, speaker2, api_key, progress_callback))
        print("✅ Podcast generated successfully!")
        sys.stdout.flush()
        return result
        
    except Exception as e:
        error_details = traceback.format_exc()
        print("\n" + "="*60)
        print("❌ ERROR OCCURRED IN PODCAST GENERATION")
        print("="*60)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("-"*60)
        print("Full Traceback:")
        print(error_details)
        print("="*60 + "\n")
        sys.stdout.flush()
        raise gr.Error(f"Error: {str(e)}")


def main(debug=True):
    load_tts_model()
    
    language_options = [
        "Auto Detect",
        # Indic Parler TTS supported languages (with available speakers)
        "Hindi", "Bengali", "Telugu", "Tamil", "Kannada", "Malayalam", "Marathi",
        "Gujarati", "Odia", "Punjabi", "Assamese", "Nepali", "Sanskrit", "English",
        "Manipuri", "Bodo", "Dogri", "Chhattisgarhi",
        # Other languages (for script generation - will use closest speaker)
        "Urdu", "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani",
        "Bahasa Indonesian", "Basque", "Bosnian", "Bulgarian", "Burmese", "Catalan",
        "Chinese Cantonese", "Chinese Mandarin", "Croatian", "Czech", "Danish", "Dutch",
        "Estonian", "Filipino", "Finnish", "French", "Georgian", "German", "Greek",
        "Hebrew", "Hungarian", "Icelandic", "Irish", "Italian", "Japanese", "Korean",
        "Latvian", "Lithuanian", "Malay", "Norwegian", "Persian", "Polish", "Portuguese",
        "Romanian", "Russian", "Serbian", "Slovak", "Spanish", "Swedish", "Thai",
        "Turkish", "Ukrainian", "Vietnamese", "Welsh"
    ]
    
    voice_options = list(VOICE_CONFIGS.keys())
    
    # Function to generate script only
    def generate_script_only(input_text, input_file, language, api_key, progress=gr.Progress()):
        if not input_text and input_file is None:
            raise gr.Error("❌ Please enter some text OR upload a file!")
        if not api_key or not api_key.strip():
            raise gr.Error("❌ Please enter your Gemini API key!")
        
        try:
            print("🚀 Generating script...")
            sys.stdout.flush()
            
            file_obj = input_file if input_file else None
            
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            podcast_gen = PodcastGenerator()
            script_json = asyncio.run(podcast_gen.generate_script(input_text, language, api_key, file_obj))
            
            # Format script for display
            script_lines = []
            for item in script_json.get('podcast', []):
                speaker = f"Speaker {item['speaker']}"
                script_lines.append(f"{speaker}: {item['line']}")
            
            formatted_script = "\n\n".join(script_lines)
            print("✅ Script generated!")
            sys.stdout.flush()
            return formatted_script
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"❌ Error generating script:\n{error_details}")
            sys.stdout.flush()
            raise gr.Error(f"Error: {str(e)}")
    
    # Function to generate audio from script
    def generate_audio_from_script(script_text, speaker1, speaker2, 
                                   s1_pace, s1_pitch, s1_tone, s1_quality,
                                   s2_pace, s2_pitch, s2_tone, s2_quality,
                                   progress=gr.Progress()):
        if not script_text or not script_text.strip():
            raise gr.Error("❌ No script to convert! Generate or enter a script first.")
        
        # Build dynamic voice descriptions
        def build_voice_description(speaker_name, pace, pitch, tone, quality):
            name = speaker_name.split(" (")[0] if " (" in speaker_name else speaker_name
            pitch_desc = f"{pitch}-pitched" if pitch != "normal" else "balanced"
            tone_desc = tone if tone != "monotone" else "slightly monotone"
            return f"{name} speaks at a {pace} pace with a {pitch_desc} voice. The delivery is {tone_desc}. {quality}."
        
        try:
            print("🎙️ Converting script to audio...")
            sys.stdout.flush()
            
            # Parse the script - handle multiple formats
            podcast_json = {"podcast": []}
            
            # Split by lines (handle both \n\n and \n)
            raw_lines = script_text.strip().replace('\r\n', '\n').split('\n')
            
            # Combine lines that belong together (if they don't start with a speaker indicator)
            combined_lines = []
            current_line = ""
            
            for raw_line in raw_lines:
                raw_line = raw_line.strip()
                if not raw_line:
                    if current_line:
                        combined_lines.append(current_line)
                        current_line = ""
                    continue
                
                # Check if this line starts with a speaker indicator
                import re
                speaker_pattern = r'^(Speaker\s*[12]|S[12]|[12])\s*[:\.]\s*'
                if re.match(speaker_pattern, raw_line, re.IGNORECASE):
                    if current_line:
                        combined_lines.append(current_line)
                    current_line = raw_line
                else:
                    # Continuation of previous line
                    if current_line:
                        current_line += " " + raw_line
                    else:
                        current_line = raw_line
            
            if current_line:
                combined_lines.append(current_line)
            
            # Parse each combined line
            for line in combined_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try different speaker patterns
                speaker = None
                text = line
                
                # Pattern: "Speaker 1:" or "Speaker 2:"
                if re.match(r'^Speaker\s*1\s*[:\.]\s*', line, re.IGNORECASE):
                    speaker = 1
                    text = re.sub(r'^Speaker\s*1\s*[:\.]\s*', '', line, flags=re.IGNORECASE)
                elif re.match(r'^Speaker\s*2\s*[:\.]\s*', line, re.IGNORECASE):
                    speaker = 2
                    text = re.sub(r'^Speaker\s*2\s*[:\.]\s*', '', line, flags=re.IGNORECASE)
                # Pattern: "S1:" or "S2:"
                elif re.match(r'^S1\s*[:\.]\s*', line, re.IGNORECASE):
                    speaker = 1
                    text = re.sub(r'^S1\s*[:\.]\s*', '', line, flags=re.IGNORECASE)
                elif re.match(r'^S2\s*[:\.]\s*', line, re.IGNORECASE):
                    speaker = 2
                    text = re.sub(r'^S2\s*[:\.]\s*', '', line, flags=re.IGNORECASE)
                # Pattern: "1:" or "2:" or "1." or "2."
                elif re.match(r'^1\s*[:\.]\s*', line):
                    speaker = 1
                    text = re.sub(r'^1\s*[:\.]\s*', '', line)
                elif re.match(r'^2\s*[:\.]\s*', line):
                    speaker = 2
                    text = re.sub(r'^2\s*[:\.]\s*', '', line)
                
                if speaker and text.strip():
                    podcast_json["podcast"].append({
                        "speaker": speaker,
                        "line": text.strip()
                    })
            
            if not podcast_json["podcast"]:
                raise gr.Error("❌ Could not parse script. Use format like:\nSpeaker 1: Hello\nSpeaker 2: Hi there\n\nOr: S1: Hello / S2: Hi / 1: Hello / 2: Hi")
            
            # Print what was parsed
            print(f"📋 Parsed {len(podcast_json['podcast'])} lines:")
            for i, item in enumerate(podcast_json['podcast'][:5]):  # Show first 5
                print(f"   {i+1}. Speaker {item['speaker']}: {item['line'][:50]}...")
            if len(podcast_json['podcast']) > 5:
                print(f"   ... and {len(podcast_json['podcast']) - 5} more lines")
            sys.stdout.flush()
            
            # Build voice descriptions with controls
            speaker1_desc = build_voice_description(speaker1, s1_pace, s1_pitch, s1_tone, s1_quality)
            speaker2_desc = build_voice_description(speaker2, s2_pace, s2_pitch, s2_tone, s2_quality)
            
            print(f"🔊 Speaker 1: {speaker1_desc}")
            print(f"🔊 Speaker 2: {speaker2_desc}")
            sys.stdout.flush()
            
            podcast_gen = PodcastGenerator()
            audio_files = []
            total_lines = len(podcast_json['podcast'])
            
            for i, item in enumerate(podcast_json['podcast']):
                print(f"📊 Progress: {int((i+1)/total_lines*100)}% - Generating speech {i+1}/{total_lines}...")
                sys.stdout.flush()
                progress((i+1)/total_lines, f"Generating speech {i+1}/{total_lines}...")
                
                audio_file = podcast_gen.tts_generate(item['line'], item['speaker'], speaker1_desc, speaker2_desc)
                audio_files.append(audio_file)
            
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            combined = asyncio.run(podcast_gen.combine_audio_files(audio_files))
            print("✅ Podcast audio generated!")
            sys.stdout.flush()
            return combined
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"❌ Error generating audio:\n{error_details}")
            sys.stdout.flush()
            raise gr.Error(f"Error: {str(e)}")
    
    with gr.Blocks(title="🎙️ PodcastGen with Indic Parler TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ PodcastGen with Indic Parler TTS")
        gr.Markdown("Generate AI podcasts with **high-quality Indian language TTS** powered by GPU!")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="📝 Input Text / Topic", lines=6, placeholder="Enter text or topic for podcast generation...")
            with gr.Column(scale=1):
                input_file = gr.File(label="📄 Or Upload File", file_types=[".pdf", ".txt"])
        
        with gr.Row():
            api_key = gr.Textbox(label="🔑 Gemini API Key", placeholder="Enter API key", type="password", scale=2)
            language = gr.Dropdown(label="🌍 Language", choices=language_options, value="Auto Detect", scale=1)
        
        # Step 1: Generate Script
        gr.Markdown("### Step 1: Generate Script")
        generate_script_btn = gr.Button("📝 Generate Script", variant="secondary", size="lg")
        
        script_output = gr.Textbox(
            label="📜 Generated Script (Edit if needed)", 
            lines=15, 
            placeholder="Script will appear here. You can edit it before generating audio.\n\nFormat:\nSpeaker 1: First speaker's line\n\nSpeaker 2: Second speaker's line",
            interactive=True
        )
        
        # Step 2: Generate Audio
        gr.Markdown("### Step 2: Choose Voices & Generate Audio")
        with gr.Row():
            speaker1 = gr.Dropdown(label="🎤 Speaker 1 Voice", choices=voice_options, value="Rohit (Hindi)")
            speaker2 = gr.Dropdown(label="🎤 Speaker 2 Voice", choices=voice_options, value="Divya (Hindi)")
        
        # Voice control options
        pace_options = ["slow", "moderate", "fast"]
        pitch_options = ["low", "normal", "high"]
        tone_options = ["monotone", "expressive", "cheerful", "urgent"]
        quality_options = ["very clear audio", "clear audio"]
        
        # Speaker 1 Voice Controls
        with gr.Accordion("🎚️ Speaker 1 Voice Settings", open=False):
            with gr.Row():
                s1_pace = gr.Dropdown(label="Pace", choices=pace_options, value="moderate")
                s1_pitch = gr.Dropdown(label="Pitch", choices=pitch_options, value="normal")
            with gr.Row():
                s1_tone = gr.Dropdown(label="Tone", choices=tone_options, value="expressive")
                s1_quality = gr.Dropdown(label="Audio Quality", choices=quality_options, value="very clear audio")
        
        # Speaker 2 Voice Controls
        with gr.Accordion("🎚️ Speaker 2 Voice Settings", open=False):
            with gr.Row():
                s2_pace = gr.Dropdown(label="Pace", choices=pace_options, value="moderate")
                s2_pitch = gr.Dropdown(label="Pitch", choices=pitch_options, value="normal")
            with gr.Row():
                s2_tone = gr.Dropdown(label="Tone", choices=tone_options, value="expressive")
                s2_quality = gr.Dropdown(label="Audio Quality", choices=quality_options, value="very clear audio")
        
        generate_audio_btn = gr.Button("🎧 Generate Podcast Audio", variant="primary", size="lg")
        output_audio = gr.Audio(label="🎧 Generated Podcast", type="filepath", format="wav")
        
        gr.Markdown("---\n### ℹ️ Tips:\n- Expand voice settings to customize pace, pitch, tone\n- Select language to auto-set recommended speakers")
        
        # Function to update speakers when language changes
        def update_speakers_for_language(lang):
            if lang in LANGUAGE_SPEAKERS:
                return LANGUAGE_SPEAKERS[lang]["speaker1"], LANGUAGE_SPEAKERS[lang]["speaker2"]
            return "Rohit (Hindi)", "Divya (Hindi)"
        
        # Connect language change to update speakers
        language.change(
            fn=update_speakers_for_language,
            inputs=[language],
            outputs=[speaker1, speaker2]
        )
        
        # Connect buttons
        generate_script_btn.click(
            fn=generate_script_only, 
            inputs=[input_text, input_file, language, api_key], 
            outputs=[script_output]
        )
        generate_audio_btn.click(
            fn=generate_audio_from_script, 
            inputs=[script_output, speaker1, speaker2, s1_pace, s1_pitch, s1_tone, s1_quality, s2_pace, s2_pitch, s2_tone, s2_quality], 
            outputs=[output_audio]
        )
    
    demo.launch(share=True, debug=debug)


if __name__ == "__main__":
    main()
