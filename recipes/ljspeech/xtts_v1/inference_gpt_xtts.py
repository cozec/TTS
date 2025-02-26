import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager

# Add safe globals for model loading
torch.serialization.add_safe_globals([
    XttsConfig,
    Xtts,
    SpeakerManager,
    dict,
    type(None),  # NoneType
])

# Add here the xtts_config path
CONFIG_PATH = "recipes/ljspeech/xtts_v1/run/training/GPT_XTTS_LJSpeech_FT-February-25-2025_08+13PM-dbf1a08a/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "recipes/ljspeech/xtts_v1/run/training/XTTS_v1.1_original_model_files"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "recipes/ljspeech/xtts_v1/run/training/GPT_XTTS_LJSpeech_FT-February-25-2025_08+13PM-dbf1a08a/checkpoint_359.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "LjSpeech_reference.wav"

# output wav path
OUTPUT_WAV_PATH = "xtts-ft.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)

# Load the model checkpoint
model.load_checkpoint(
    config, 
    checkpoint_path=XTTS_CHECKPOINT, 
    vocab_path=TOKENIZER_PATH, 
    use_deepspeed=False,
    speaker_file_path=SPEAKER_REFERENCE
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)