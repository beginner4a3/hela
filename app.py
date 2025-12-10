import gradio as gr
from pydub import AudioSegment
from google import genai
from google.genai import types
import json
import uuid
import asyncio
import aiofiles
import os
import time
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


# Voice configurations for Indic Parler TTS
VOICE_CONFIGS = {
    # Male Voices
    "Rohit - Male (Hindi/English)": "Rohit speaks in a clear, moderate pace with a confident tone. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    "Karan - Male (Hindi/English)": "Karan's voice is warm and expressive with a moderate speed and natural pitch. The recording is very high quality with no background noise.",
    "Vikram - Male (Hindi/English)": "Vikram speaks in a deep, slightly expressive tone with moderate speed. Very clear audio with professional quality.",
    "Arjun - Male (Telugu)": "Arjun delivers speech with a clear, conversational tone at moderate pace. High quality recording with no background noise.",
    "Suresh - Male (Tamil)": "Suresh speaks with a warm, engaging tone at moderate speed. The recording is of very high quality.",
    
    # Female Voices  
    "Divya - Female (Hindi/English)": "Divya's voice is expressive and animated with a moderate speed and pleasant pitch. The recording is very high quality with the speaker's voice sounding clear.",
    "Leela - Female (Hindi/English)": "Leela speaks in a high-pitched, fast-paced, and cheerful tone, full of energy and happiness. The recording is very high quality with no background noise.",
    "Maya - Female (Hindi/English)": "Maya's voice is soft and soothing with a calm, measured delivery. Very clear audio with professional recording quality.",
    "Sita - Female (Telugu)": "Sita speaks with a melodious, expressive tone at moderate pace. High quality recording with clear audio.",
    "Priya - Female (Tamil)": "Priya delivers speech with a warm, conversational tone. The recording is very high quality with no background noise.",
    
    # Accent-specific voices
    "Indian English Male": "A male Indian English speaker with a clear accent delivers slightly expressive speech at moderate speed. Very high quality recording with clear audio.",
    "Indian English Female": "A female Indian English speaker with a pleasant accent delivers animated speech with moderate pitch. The recording is of very high quality.",
}


class PodcastGenerator:
    def __init__(self):
        pass

    async def generate_script(self, prompt: str, language: str, api_key: str, file_obj=None, progress=None) -> Dict:
        example = """
{
    "topic": "AGI",
    "podcast": [
        {
            "speaker": 2,
            "line": "So, AGI, huh? Seems like everyone's talking about it these days."
        },
        {
            "speaker": 1,
            "line": "Yeah, it's definitely having a moment, isn't it?"
        },
        {
            "speaker": 2,
            "line": "It is and for good reason, right? I mean, you've been digging into this stuff, listening to the podcasts and everything. What really stood out to you? What got you hooked?"
        },
        {
            "speaker": 1,
            "line": "Honestly, it's the sheer scale of what AGI could do. We're talking about potentially reshaping well everything."
        },
        {
            "speaker": 2,
            "line": "No kidding, but let's be real. Sometimes it feels like every other headline is either hyping AGI up as this technological utopia or painting it as our inevitable robot overlords."
        },
        {
            "speaker": 1,
            "line": "It's easy to get lost in the noise, for sure."
        },
        {
            "speaker": 2,
            "line": "Exactly. So how about we try to cut through some of that, shall we?"
        },
        {
            "speaker": 1,
            "line": "Sounds like a plan."
        }
    ]
}
        """

        if language == "Auto Detect":
            language_instruction = "- The podcast MUST be in the same language as the user input."
        else:
            language_instruction = f"- The podcast MUST be in {language} language"

        system_prompt = f"""
You are a professional podcast generator. Your task is to generate a professional podcast script based on the user input.
{language_instruction}
- The podcast should have 2 speakers.
- The podcast should be long.
- Do not use names for the speakers.
- The podcast should be interesting, lively, and engaging, and hook the listener from the start.
- The input text might be disorganized or unformatted, originating from sources like PDFs or text files. Ignore any formatting inconsistencies or irrelevant details; your task is to distill the essential points, identify key definitions, and highlight intriguing facts that would be suitable for discussion in a podcast.
- The script must be in JSON format.
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
                    parts=[
                        types.Part.from_bytes(
                            data=file_data,
                            mime_type=mime_type,
                        )
                    ],
                )
            )
        
        messages.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_prompt)
                ],
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
                    model="gemini-2.0-flash",
                    contents=messages,
                    config=types.GenerateContentConfig(
                        temperature=1,
                        response_mime_type="application/json",
                        safety_settings=[
                            types.SafetySetting(
                                category=safety_setting["category"],
                                threshold=safety_setting["threshold"]
                            ) for safety_setting in safety_settings
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
                raise Exception("Rate limit exceeded for the API key. Please try again later or provide your own Gemini API key.")
            else:
                raise Exception(f"Failed to generate podcast script: {e}")

        print(f"Generated podcast script:\n{response.text}")
        
        if progress:
            progress(0.4, "Script generated successfully!")
            
        return json.loads(response.text)
    
    async def _read_file_bytes(self, file_obj) -> bytes:
        """Read file bytes from a file object"""
        if hasattr(file_obj, 'size'):
            file_size = file_obj.size
        else:
            file_size = os.path.getsize(file_obj.name)
            
        if file_size > MAX_FILE_SIZE_BYTES:
            raise Exception(f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit. Please upload a smaller file.")
            
        if hasattr(file_obj, 'read'):
            return file_obj.read()
        else:
            async with aiofiles.open(file_obj.name, 'rb') as f:
                return await f.read()
    
    def _get_mime_type(self, filename: str) -> str:
        """Determine MIME type based on file extension"""
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return "application/pdf"
        elif ext == '.txt':
            return "text/plain"
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            return mime_type or "application/octet-stream"

    def tts_generate(self, text: str, speaker: int, speaker1_desc: str, speaker2_desc: str) -> str:
        """Generate TTS using Indic Parler TTS (GPU accelerated)"""
        voice_desc = speaker1_desc if speaker == 1 else speaker2_desc
        
        # Tokenize description and prompt
        description_input_ids = DESCRIPTION_TOKENIZER(
            voice_desc, 
            return_tensors="pt"
        ).to(DEVICE)
        
        prompt_input_ids = TOKENIZER(
            text, 
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate audio
        with torch.no_grad():
            generation = MODEL.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask
            )
        
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Save to temp file
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
                timeout=1800  # 30 minutes for GPU TTS
            )
        except asyncio.TimeoutError:
            raise Exception("The podcast generation process timed out. Please try with shorter text or try again later.")
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
        
        # Process TTS sequentially (GPU memory optimization)
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
        
        combined_audio = await self.combine_audio_files(audio_files, progress)
        return combined_audio


async def process_input(input_text: str, input_file, language: str, speaker1: str, speaker2: str, api_key: str = "", progress=None) -> str:
    start_time = time.time()

    # Get voice descriptions from VOICE_CONFIGS
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

        end_time = time.time()
        print(f"Total podcast generation time: {end_time - start_time:.2f} seconds")
        return podcast
        
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            raise Exception("Rate limit exceeded. Please try again later or use your own API key.")
        elif "timeout" in error_msg.lower():
            raise Exception("The request timed out. This could be due to server load or the length of your input. Please try again with shorter text.")
        else:
            raise Exception(f"Error: {error_msg}")


# Gradio UI
def generate_podcast_gradio(input_text, input_file, language, speaker1, speaker2, api_key, progress=gr.Progress()):
    file_obj = None
    if input_file is not None:
        file_obj = input_file
        
    def progress_callback(value, text):
        progress(value, text)

    result = asyncio.run(process_input(
        input_text, 
        file_obj, 
        language, 
        speaker1, 
        speaker2, 
        api_key,
        progress_callback
    ))
    
    return result


def main():
    # Load TTS model at startup
    load_tts_model()
    
    # Language options (Indian languages first)
    language_options = [
        "Auto Detect",
        # Indian Languages (supported by Indic Parler TTS)
        "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Marathi",
        "Gujarati", "Odia", "Punjabi", "Assamese", "Urdu", "Nepali", "Sanskrit",
        "English",
        # Other languages
        "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani",
        "Bahasa Indonesian", "Bangla", "Basque", "Bosnian", "Bulgarian",
        "Burmese", "Catalan", "Chinese Cantonese", "Chinese Mandarin",
        "Chinese Taiwanese", "Croatian", "Czech", "Danish", "Dutch",
        "Estonian", "Filipino", "Finnish", "French", "Galician", "Georgian",
        "German", "Greek", "Hebrew", "Hungarian", "Icelandic", "Irish",
        "Italian", "Japanese", "Javanese", "Kazakh", "Khmer", "Korean",
        "Lao", "Latvian", "Lithuanian", "Macedonian", "Malay",
        "Maltese", "Mongolian", "Norwegian Bokmål", "Pashto", "Persian",
        "Polish", "Portuguese", "Romanian", "Russian", "Serbian", "Sinhala",
        "Slovak", "Slovene", "Somali", "Spanish", "Sundanese", "Swahili",
        "Swedish", "Thai", "Turkish", "Ukrainian", "Uzbek", "Vietnamese", "Welsh", "Zulu"
    ]
    
    # Voice options from VOICE_CONFIGS
    voice_options = list(VOICE_CONFIGS.keys())
    
    # Create Gradio interface
    with gr.Blocks(title="🎙️ PodcastGen with Indic Parler TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ PodcastGen with Indic Parler TTS")
        gr.Markdown("""Generate AI podcasts with **high-quality Indian language TTS** powered by GPU!
        
Supports: Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, and 21+ languages!""")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="📝 Input Text", 
                    lines=10, 
                    placeholder="Enter text for podcast generation...\n\nYou can enter text in Hindi, Telugu, Tamil, or any supported language!"
                )
            
            with gr.Column(scale=1):
                input_file = gr.File(label="📄 Or Upload a PDF or TXT file", file_types=[".pdf", ".txt"])
        
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="🔑 Your Gemini API Key", placeholder="Enter API key here", type="password")
                language = gr.Dropdown(label="🌍 Language", choices=language_options, value="Auto Detect")

            with gr.Column():
                speaker1 = gr.Dropdown(label="🎤 Speaker 1 Voice", choices=voice_options, value="Rohit - Male (Hindi/English)")
                speaker2 = gr.Dropdown(label="🎤 Speaker 2 Voice", choices=voice_options, value="Divya - Female (Hindi/English)")
        
        generate_btn = gr.Button("🚀 Generate Podcast", variant="primary", size="lg")
        
        with gr.Row():
            output_audio = gr.Audio(label="🎧 Generated Podcast", type="filepath", format="wav")
        
        gr.Markdown("""---
### ℹ️ Tips:
- The model **automatically detects** the language from your input text
- For best results with Indian languages, choose matching speaker voices  
- GPU acceleration makes TTS generation much faster
- First run may take longer as the model warms up""")
            
        generate_btn.click(
            fn=generate_podcast_gradio,
            inputs=[input_text, input_file, language, speaker1, speaker2, api_key],
            outputs=[output_audio]
        )
    
    demo.launch(share=True)


if __name__ == "__main__":
    main()