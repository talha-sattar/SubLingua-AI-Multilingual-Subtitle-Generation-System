# Yam_SubG.py
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub # Make sure to import hub
import assemblyai as aai
from typing import List, Dict, Set, Optional
try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI library not found. Translation features will be disabled.")
    print("Install using: pip install openai")
    OpenAI = None # Set to None if not available

import pandas as pd
from typing import List, Dict, Optional, Set, Tuple, Any # Added Any
from dataclasses import dataclass, replace as dc_replace # Use dataclasses.replace
import logging
import warnings
from tqdm import tqdm
import librosa # Ensure librosa is installed: pip install librosa
import json # Added json import

# ############################################################################
# # WARNING: HARDCODED API KEYS & PATHS - MAJOR SECURITY RISK & POOR PRACTICE #
# ############################################################################
# It is STRONGLY recommended to use environment variables (os.environ.get)
# or a dedicated configuration management system (e.g., config files, Vault)
# instead of hardcoding sensitive information directly in the source code.
# Pushing this code to public repositories WILL expose your keys.

# --- Hardcoded Configuration ---
HARDCODED_ASSEMBLYAI_API_KEY = "adfadf7f34654a4fa01e2ff7d07d563a"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WARNING: Hardcoded paths are not flexible. Consider arguments or config.
PROCESSING_DIR = "processing_steps"
OUTPUT_DIR = "output"
# IMPORTANT: MODEL_DIR should point to the *directory containing the SavedModel files*
# (e.g., the directory with saved_model.pb and variables/ folder),
# NOT the .pb file itself. Leave empty or None if only using TF Hub.
MODEL_DIR = r"C:\Users\tsatt\Downloads\Music\YAMnet\Model"
# Path to the class map CSV file. Should be inside MODEL_DIR or specified separately.
CLASS_MAP_CSV = "Model\yamnet_class_map.csv" # Handle case where MODEL_DIR is None
# --- End Hardcoded Configuration ---
# ############################################################################


# --- TensorFlow/Warnings Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
# --- End Configuration ---

# --- Constants and Configuration Class ---
class Config:
    """Holds configuration parameters for the subtitle generation process."""
    TARGET_SR = 16000 # Required sample rate for YAMNet
    CHUNK_DURATION_SEC = 0.975 # Duration corresponding to YAMNet's expected input window
    EXPECTED_SAMPLES = int(TARGET_SR * CHUNK_DURATION_SEC) # 15600 samples

    # Confidence thresholds
    NONVERBAL_CONFIDENCE_THRESHOLD = 0.5
    STANDALONE_CONFIDENCE_THRESHOLD = 0.4

    # Subtitle timing constraints (in milliseconds)
    MIN_SUBTITLE_DURATION_MS = 1000
    MAX_SUBTITLE_DURATION_MS = 7000
    MIN_GAP_MS = 150
    MAX_OVERLAP_MS = 300 # Max overlap for merging NV into V

    # Subtitle formatting
    MAX_CHARS_PER_LINE = 42
    MAX_LINES_PER_SUBTITLE = 2

    # API settings
    ASSEMBLYAI_RETRY_ATTEMPTS = 3
    ASSEMBLYAI_RETRY_DELAY_SEC = 5
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS_PER_SUB = 100
    OPENAI_RETRY_ATTEMPTS = 2
    OPENAI_RETRY_DELAY_SEC = 3

# --- Logging Configuration ---
LOG_FILE = "subtitle_generator.log"
if os.path.exists(LOG_FILE):
    try: os.remove(LOG_FILE)
    except OSError as e: print(f"Warning: Could not clear previous log file: {e}")

logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose output
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[ logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler() ]
)
# --- End Logging Configuration ---

# --- YAMNet Label Definitions ---
# (Keep the extensive YAMNET_LABEL_MAP, GENRE_SOUND_FILTERS, OBSCURE_LABELS dictionaries here - unchanged)
YAMNET_LABEL_MAP = {
    "Speech": "[Speech]", "Child speech, kid speaking": "[Child speaking]", "Conversation": "[Conversation]", "Narration, monologue": "[Narrator]", "Babbling": "[Babbling]", "Speech synthesizer": "[Synthesized speech]", "Shout": "[Shouting]", "Bellow": "[Bellowing]", "Whoop": "[Whooping]", "Yell": "[Yelling]", "Children shouting": "[Children shouting]", "Screaming": "[Screaming]", "Whispering": "[Whispering]", "Laughter": "[Laughter]", "Baby laughter": "[Baby laughing]", "Giggle": "[Giggling]", "Snicker": "[Snickering]", "Belly laugh": "[Belly laugh]", "Chuckle, chortle": "[Chuckling]", "Crying, sobbing": "[Crying]", "Baby cry, infant cry": "[Baby crying]", "Whimper": "[Whimpering]", "Wail, moan": "[Wailing]", "Sigh": "[Sighing]", "Singing": "[Singing]", "Choir": "[Choir singing]", "Yodeling": "[Yodeling]", "Chant": "[Chanting]", "Mantra": "[Chanting]", "Child singing": "[Child singing]", "Synthetic singing": "[Synthesized singing]", "Rapping": "[Rapping]", "Humming": "[Humming]", "Groan": "[Groaning]", "Grunt": "[Grunting]", "Whistling": "[Whistling]", "Breathing": "[Breathing]", "Gasp": "[Gasping]", "Pant": "[Panting]", "Snore": "[Snoring]", "Cough": "[Coughing]", "Sneeze": "[Sneezing]", "Sniff": "[Sniffing]", "Run": "[Running sounds]", "Shuffle": "[Shuffling]", "Footsteps": "[Footsteps]", "Chewing, mastication": "[Chewing]", "Biting": "[Biting]", "Gargling": "[Gargling]", "Burping, eructation": "[Burping]", "Hiccup": "[Hiccup]", "Fart": "[Farting]", "Hands": "[Hand sounds]", "Finger snapping": "[Finger snapping]", "Clapping": "[Clapping]", "Heart sounds, heartbeat": "[Heartbeat]", "Respiratory sounds": "[Breathing]", "Human group actions": "[Group sounds]", "Cheering": "[Cheering]", "Applause": "[Applause]", "Chatter": "[Chatter]", "Crowd": "[Crowd noise]", "Hubbub, speech noise, speech babble": "[Babble]", "Animal": "[Animal sounds]", "Domestic animals, pets": "[Pet sounds]", "Dog": "[Dog barking]", "Bark": "[Dog barking]", "Yip": "[Dog yipping]", "Howl": "[Howling]", "Bow-wow": "[Dog barking]", "Growling": "[Growling]", "Whimper (dog)": "[Dog whimpering]", "Cat": "[Cat meowing]", "Purr": "[Purring]", "Meow": "[Meowing]", "Hiss": "[Hissing]", "Caterwaul": "[Cat yowling]", "Livestock, farm animals, working animals": "[Livestock sounds]", "Horse": "[Horse sounds]", "Clip-clop": "[Hoofbeats]", "Neigh, whinny": "[Horse neighing]", "Cattle, bovinae": "[Cow sounds]", "Moo": "[Mooing]", "Cowbell": "[Cowbell]", "Pig": "[Pig sounds]", "Oink": "[Oinking]", "Goat": "[Goat sounds]", "Bleat": "[Bleating]", "Sheep": "[Sheep sounds]", "Fowl": "[Fowl sounds]", "Chicken, rooster": "[Chicken clucking]", "Cluck": "[Clucking]", "Crowing, cock-a-doodle-doo": "[Rooster crowing]", "Turkey": "[Turkey gobbling]", "Gobble": "[Gobbling]", "Duck": "[Duck quacking]", "Quack": "[Quacking]", "Goose": "[Goose honking]", "Honk": "[Honking]", "Wild animals": "[Wild animal sounds]", "Roaring cats (lions, tigers)": "[Roaring]", "Roar": "[Roaring]", "Bird": "[Bird chirping]", "Bird vocalization, bird call, bird song": "[Birdsong]", "Chirp, tweet": "[Chirping]", "Squawk": "[Squawking]", "Pigeon, dove": "[Dove cooing]", "Coo": "[Cooing]", "Crow": "[Crow cawing]", "Caw": "[Cawing]", "Owl": "[Owl hooting]", "Hoot": "[Hooting]", "Insect": "[Insect buzzing]", "Cricket": "[Cricket chirping]", "Mosquito": "[Mosquito buzzing]", "Fly, housefly": "[Fly buzzing]", "Buzz": "[Buzzing]", "Bee, wasp, etc.": "[Bee buzzing]", "Frog": "[Frog croaking]", "Croak": "[Croaking]", "Snake": "[Snake hissing]", "Rattle": "[Rattling]", "Whale vocalization": "[Whale sounds]", "Music": "[Music]", "Musical instrument": "[Instrumental music]", "Plucked string instrument": "[String instrument]", "Guitar": "[Guitar music]", "Acoustic guitar": "[Acoustic guitar]", "Steel guitar, slide guitar": "[Slide guitar]", "Electric guitar": "[Electric guitar]", "Banjo": "[Banjo music]", "Sitar": "[Sitar music]", "Mandolin": "[Mandolin music]", "Zither": "[Zither music]", "Ukulele": "[Ukulele music]", "Keyboard (musical)": "[Keyboard music]", "Piano": "[Piano music]", "Electric piano": "[Electric piano]", "Organ": "[Organ music]", "Electronic organ": "[Electronic organ]", "Hammond organ": "[Hammond organ]", "Synthesizer": "[Synthesizer music]", "Sampler": "[Sampler music]", "Harpsichord": "[Harpsichord music]", "Percussion": "[Percussion]", "Drum kit": "[Drum kit]", "Drum machine": "[Drum machine]", "Drum": "[Drum beat]", "Snare drum": "[Snare drum]", "Rimshot": "[Rimshot]", "Drum stick": "[Drum stick sounds]", "Bass drum": "[Bass drum]", "Timpani": "[Timpani]", "Tabla": "[Tabla]", "Cymbal": "[Cymbal crash]", "Hi-hat": "[Hi-hat cymbal]", "Wood block": "[Wood block]", "Tambourine": "[Tambourine]", "Rattle (instrument)": "[Rattle shaking]", "Maraca": "[Maracas]", "Gong": "[Gong]", "Mallet percussion": "[Mallet percussion]", "Marimba, xylophone": "[Xylophone]", "Glockenspiel": "[Glockenspiel]", "Vibraphone": "[Vibraphone]", "Steelpan": "[Steel drum]", "Triangle": "[Triangle ding]", "Bell": "[Bell ringing]", "Church bell": "[Church bell]", "Jingle bell": "[Jingle bells]", "Bicycle bell": "[Bicycle bell]", "Tuning fork": "[Tuning fork]", "Chime": "[Chimes]", "Wind chime": "[Wind chimes]", "Change ringing (campanology)": "[Change ringing]", "Wound string instrument": "[Bowed string instrument]", "Violin, fiddle": "[Violin music]", "Viola": "[Viola music]", "Cello": "[Cello music]", "Double bass": "[Double bass music]", "Wind instrument, woodwind instrument": "[Wind instrument]", "Flute": "[Flute music]", "Piccolo": "[Piccolo music]", "Recorder": "[Recorder music]", "Oboe": "[Oboe music]", "Clarinet": "[Clarinet music]", "Bassoon": "[Bassoon music]", "Saxophone": "[Saxophone music]", "Bagpipes": "[Bagpipes]", "Brass instrument": "[Brass instrument]", "French horn": "[French horn]", "Trumpet": "[Trumpet music]", "Trombone": "[Trombone music]", "Tuba": "[Tuba music]", "Cornet": "[Cornet music]", "Bugle": "[Bugle call]", "Harmonica": "[Harmonica]", "Accordion": "[Accordion music]", "Scratching (performance technique)": "[DJ scratching]", "Pop music": "[Pop music]", "Hip hop music": "[Hip hop music]", "Rock music": "[Rock music]", "Heavy metal": "[Heavy metal music]", "Punk rock": "[Punk rock music]", "Grunge": "[Grunge music]", "Progressive rock": "[Progressive rock]", "Rock and roll": "[Rock and roll]", "Psychedelic rock": "[Psychedelic rock]", "Rhythm and blues": "[R&B music]", "Soul music": "[Soul music]", "Reggae": "[Reggae music]", "Country": "[Country music]", "Swing music": "[Swing music]", "Bluegrass": "[Bluegrass music]", "Funk": "[Funk music]", "Folk music": "[Folk music]", "Middle Eastern music": "[Middle Eastern music]", "Jazz": "[Jazz music]", "Disco": "[Disco music]", "Classical music": "[Classical music]", "Opera": "[Opera music]", "Electronic music": "[Electronic music]", "House music": "[House music]", "Techno": "[Techno music]", "Dubstep": "[Dubstep music]", "Drum and bass": "[Drum and bass]", "Electronica": "[Electronica music]", "Electronic dance music": "[EDM]", "Ambient music": "[Ambient music]", "Trance music": "[Trance music]", "Music of Latin America": "[Latin American music]", "Salsa music": "[Salsa music]", "Flamenco": "[Flamenco music]", "Blues": "[Blues music]", "Music for children": "[Children's music]", "New-age music": "[New-age music]", "Vocal music": "[Vocal music]", "A capella": "[A capella singing]", "Music of Africa": "[African music]", "Afrobeat": "[Afrobeat music]", "Music of Asia": "[Asian music]", "Indian music": "[Indian music]", "Bollywood": "[Bollywood music]", "Ska": "[Ska music]", "Traditional music": "[Traditional music]", "Independent music": "[Indie music]", "Theme music": "[Theme music]", "Jingle (music)": "[Jingle]", "Soundtrack music": "[Soundtrack music]", "Lullaby": "[Lullaby]", "Video game music": "[Video game music]", "Christmas music": "[Christmas music]", "Dance music": "[Dance music]", "Wedding music": "[Wedding music]", "Happy music": "[Happy music]", "Sad music": "[Sad music]", "Tender music": "[Tender music]", "Exciting music": "[Exciting music]", "Angry music": "[Angry music]", "Scary music": "[Scary music]", "Wind": "[Wind blowing]", "Rustling leaves": "[Leaves rustling]", "Wind noise (microphone)": "[Wind noise]", "Thunderstorm": "[Thunderstorm]", "Thunder": "[Thunder rumbling]", "Water": "[Water sounds]", "Rain": "[Rain falling]", "Raindrop": "[Raindrops]", "Rain on surface": "[Rain on surface]", "Stream": "[Stream flowing]", "Waterfall": "[Waterfall]", "Ocean": "[Ocean waves]", "Waves, surf": "[Waves crashing]", "Gurgling": "[Gurgling]", "Fire": "[Fire crackling]", "Crackle": "[Crackling fire]", "Vehicle": "[Vehicle sounds]", "Car": "[Car engine]", "Vehicle horn, car horn, honking": "[Car horn]", "Car alarm": "[Car alarm]", "Tire squeal": "[Tire squeal]", "Truck": "[Truck engine]", "Air horn, truck horn": "[Truck horn]", "Reversing beeps": "[Reversing beeps]", "Police car (siren)": "[Siren wailing]", "Ambulance (siren)": "[Siren wailing]", "Fire engine, fire truck (siren)": "[Siren wailing]", "Motorcycle": "[Motorcycle engine]", "Traffic noise, roadway noise": "[Traffic noise]", "Train horn": "[Train horn]", "Aircraft": "[Aircraft noise]", "Helicopter": "[Helicopter noise]", "Engine": "[Engine running]", "Door": "[Door creaking]", "Doorbell": "[Doorbell]", "Slam": "[Door slamming]", "Knock": "[Knocking]", "Squeak": "[Squeaking]", "Cupboard open or close": "[Cupboard opening/closing]", "Drawer open or close": "[Drawer opening/closing]", "Dishes, pots, and pans": "[Clattering dishes]", "Cutlery, silverware": "[Clinking cutlery]", "Frying (food)": "[Sizzling]", "Microwave oven": "[Microwave humming]", "Blender": "[Blender whirring]", "Water tap, faucet": "[Running water]", "Toilet flush": "[Toilet flushing]", "Vacuum cleaner": "[Vacuum cleaner]", "Keys jangling": "[Keys jangling]", "Scissors": "[Scissor snip]", "Typing": "[Typing]", "Telephone bell ringing": "[Phone ringing]", "Alarm clock": "[Alarm clock]", "Siren": "[Siren wailing]", "Buzzer": "[Buzzer]", "Smoke detector, smoke alarm": "[Smoke alarm]", "Fire alarm": "[Fire alarm]", "Whistle": "[Whistle]", "Clock": "[Clock ticking]", "Tick-tock": "[Clock ticking]", "Mechanical fan": "[Fan whirring]", "Air conditioning": "[Air conditioning]", "Cash register": "[Cash register]", "Printer": "[Printer]", "Camera": "[Camera click]", "Hammer": "[Hammering]", "Jackhammer": "[Jackhammer]", "Sawing": "[Sawing]", "Drill": "[Drilling]", "Explosion": "[Explosion]", "Gunshot, gunfire": "[Gunshot]", "Fireworks": "[Fireworks]", "Wood": "[Wood creaking]", "Chop": "[Chopping]", "Glass": "[Glass breaking]", "Chink, clink": "[Clinking glass]", "Shatter": "[Glass shattering]", "Splash, splatter": "[Splashing]", "Slosh": "[Sloshing]", "Drip": "[Dripping]", "Pour": "[Pouring]", "Spray": "[Spraying]", "Boiling": "[Boiling]", "Arrow": "[Arrow whoosh]", "Whoosh, swoosh, swish": "[Whooshing]", "Thump, thud": "[Loud thud]", "Bang": "[Bang]", "Slap, smack": "[Slapping]", "Smash, crash": "[Crashing]", "Breaking": "[Breaking]", "Bouncing": "[Bouncing]", "Scratch": "[Scratching]", "Scrape": "[Scraping]", "Roll": "[Rolling]", "Crushing": "[Crushing]", "Crumpling, crinkling": "[Crumpling]", "Tearing": "[Tearing]", "Beep, bleep": "[Beeping]", "Ping": "[Ping]", "Ding": "[Ding]", "Clang": "[Clanging]", "Squeal": "[Squealing]", "Creak": "[Creaking floors]", "Rustle": "[Rustling]", "Whir": "[Whirring]", "Clatter": "[Clattering]", "Sizzle": "[Sizzling]", "Clicking": "[Clicking]", "Rumble": "[Rumbling]", "Plop": "[Plopping]", "Jingle, tinkle": "[Jingling]", "Crunch": "[Crunching]", "Silence": "[Silence]", "Domestic sounds, home sounds": "[Home sounds]", "Sound effect": "[Sound effect]", "Alarm": "[Alarm sound]",
}
GENRE_SOUND_FILTERS = { # Keep genre filters as defined by user
    "Horror": {"allow": ["Screaming", "Wailing", "Breathing", "Gasping", "Whimpering", "Groaning", "Heartbeat", "Creaking floors", "Loud thud", "Door creaking", "Knocking", "Door slamming", "Wind blowing", "Thunder rumbling", "Footsteps", "Rustling", "Scratching", "Scary music", "[Silence]"], "block": ["Laughter", "Cheering", "Applause", "Happy music", "Pop music", "Children's music"]},
    "Action": {"allow": ["Gunshot", "Explosion", "Screaming", "Shouting", "Car engine", "Tire squeal", "Crashing", "Loud thud", "Footsteps", "Siren wailing", "Helicopter noise", "Aircraft noise", "Clanging", "Breaking", "Exciting music", "Heavy metal music"], "block": ["Laughter", "Clock ticking", "Tender music", "Lullaby", "Whispering"]},
    "Romance": {"allow": ["Tender music", "Violin music", "Piano music", "Acoustic guitar", "Whispering", "Sighing", "Breathing", "Footsteps", "Rustling", "Clock ticking", "Rain falling", "Fire crackling", "Clinking glass"], "block": ["Explosion", "Gunshot", "Screaming", "Shouting", "Heavy metal music", "Scary music"]},
    "Thriller": {"allow": ["Screaming", "Gasping", "Heartbeat", "Breathing", "Whispering", "Footsteps", "Running sounds", "Door creaking", "Loud thud", "Knocking", "Squealing", "Tire squeal", "Clock ticking", "Phone ringing", "Scary music", "Exciting music", "[Silence]"], "block": ["Laughter", "Cheering", "Applause", "Happy music", "Children's music"]},
    "Comedy": {"allow": ["Laughter", "Giggling", "Snickering", "Chuckling", "Applause", "Cheering", "Footsteps", "Door creaking", "Clattering", "Bouncing", "Boing", "[Slapping]", "Happy music", "Pop music"], "block": ["Screaming", "Crying", "Explosion", "Gunshot", "Scary music", "Sad music"]},
    "Drama": {"allow": ["Crying", "Sighing", "Whispering", "Breathing", "Footsteps", "Door creaking", "Phone ringing", "Clock ticking", "Rain falling", "Thunder rumbling", "Sad music", "Tender music", "Classical music", "Piano music"], "block": ["Explosion", "Gunshot", "Laughter", "Cheering", "Happy music", "Heavy metal music"]},
    "Sci-Fi": {"allow": ["Explosion", "Gunshot", "Screaming", "[Synthesizer music]", "[Electronic music]", "[Whooshing]", "[Beeping]", "[Buzzing]", "[Rumbling]", "[Alarm sound]", "[Siren wailing]", "[Synthesized speech]", "[Engine running]", "[Footsteps]", "[Laser sounds]"], "block": ["Laughter", "Clock ticking", "Acoustic guitar", "Folk music", "[Animal sounds]", "[Bird chirping]"]},
}
OBSCURE_LABELS = {
    "Vehicle", # Added based on user feedback
    "Helicopter", # Added based on user feedback
    "Aircraft", # Might also be prone to misclassification? (Optional)
    "Engine", # Broad category, often part of music/effects (Optional)
    "Rumble", # Often part of music/effects (Optional)
    # --- Original Obscure Labels ---
    "Toot", "Power windows, electric windows", "Skidding", "Car passing by", "Air brake", "Ice cream truck, ice cream van", "Bus", "Emergency vehicle", "Rail transport", "Train", "Train whistle", "Railroad car, train wagon", "Train wheels squealing", "Subway, metro, underground", "Aircraft engine", "Jet engine", "Propeller, airscrew", "Fixed-wing aircraft, airplane", "Light engine (high frequency)", "Dental drill, dentist's drill", "Lawn mower", "Chainsaw", "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine knocking", "Engine starting", "Idling", "Accelerating, revving, vroom", "Ding-dong", "Sliding door", "Tap", "Chopping (food)", "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer", "Toothbrush", "Electric toothbrush", "Zipper (clothing)", "Coin (dropping)", "Electric shaver, electric razor", "Shuffling cards", "Typewriter", "Computer keyboard", "Writing", "Telephone", "Ringtone", "Telephone dialing, DTMF", "Dial tone", "Busy signal", "Civil defense siren", "Foghorn", "Steam whistle", "Mechanisms", "Ratchet, pawl", "Tick", "Gears", "Pulleys", "Sewing machine", "Single-lens reflex camera", "Tools", "Filing (rasp)", "Sanding", "Power tool", "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Burst, pop", "Eruption", "Boom", "Splinter", "Liquid", "Squish", "Trickle, dribble", "Gush", "Fill (with liquid)", "Pump (liquid)", "Stir", "Sonar", "Effects unit", "Chorus effect", "Basketball bounce", "Whip", "Flap", "Rub", "Clickety-clack", "Zing", "Boing", "Environmental noise", "Static", "Mains hum", "Distortion", "Sidetone", "Cacophony", "White noise", "Pink noise", "Throbbing", "Vibration", "Television", "Radio", "Field recording", "Reverberation", "Echo", "Noise", "Inside, small room", "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural", "Domestic sounds, home sounds", "Sound effect", "Alarm", "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Child singing", "Synthetic singing", "Rapping", "Humming", "Music", "Musical instrument", "Vocal music",
}
# --- End YAMNet Definitions ---


# --- Helper Functions ---
# (Keep format_timestamp and load_and_prepare_chunk functions here - unchanged)
def format_timestamp(milliseconds: Optional[float], format_type: str = 'srt') -> str:
    """Convert milliseconds to SRT or VTT timestamp format."""
    if not isinstance(milliseconds, (int, float)) or milliseconds < 0:
        milliseconds = 0 # Default to 0 if invalid input
    milliseconds = int(round(milliseconds)) # Round to nearest millisecond

    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000

    separator = ',' if format_type == 'srt' else '.'
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"

def load_and_prepare_chunk(filepath: str, target_sr: int, expected_samples: int) -> Optional[np.ndarray]:
    """Load audio chunk, resample, normalize, and ensure exact length."""
    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if np.isnan(audio).any() or np.isinf(audio).any() or np.all(audio == 0):
            logging.warning(f"Audio appears invalid (NaN, Inf, or silent) in file: {filepath}")
            return None
        if len(audio) < 10:
            logging.warning(f"Audio too short (< 10 samples) in file: {filepath}")
            return None

        max_abs = np.max(np.abs(audio))
        if max_abs > 1.0:
            logging.debug(f"Normalizing audio from max abs {max_abs:.3f} to 1.0 in {filepath}")
            audio = audio / max_abs
        elif max_abs == 0: pass # Avoid division by zero for silent audio

        current_len = len(audio)
        if current_len < expected_samples:
            pad_width = expected_samples - current_len
            audio = np.pad(audio, (0, pad_width), mode='constant')
            logging.debug(f"Padded chunk {os.path.basename(filepath)} by {pad_width} samples.")
        elif current_len > expected_samples:
            audio = audio[:expected_samples]
            logging.debug(f"Truncated chunk {os.path.basename(filepath)} from {current_len} to {expected_samples} samples.")

        return audio.astype(np.float32)
    except Exception as e:
        logging.error(f"Error loading/preparing chunk {filepath}: {e}", exc_info=True)
        return None
# --- End Helper Functions ---

# --- Dataclasses ---
@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    start_time: int # Milliseconds
    end_time: int # Milliseconds
    text: str
    type: str # 'verbal', 'non_verbal', 'combined'
    confidence: float = 0.0 # Optional: store confidence score (0.0-1.0)
# --- End Dataclasses ---

# --- Main Subtitle Generator Class ---
class SubtitleGenerator:
    """Orchestrates the subtitle generation process using YAMNet and AssemblyAI."""

    def __init__(self):
        """Initialize API clients, load models, and prepare configurations."""
        logging.info("Initializing SubtitleGenerator...")
        # --- Initialize API Clients ---
        logging.warning("Initializing API clients with hardcoded keys (SECURITY RISK!).")
        self.openai_api_key = HARDCODED_OPENAI_API_KEY
        self.assemblyai_api_key = HARDCODED_ASSEMBLYAI_API_KEY

        if not self.assemblyai_api_key:
            logging.critical("AssemblyAI API Key is missing.")
            raise ValueError("AssemblyAI API Key is missing.")
        if not self.openai_api_key and OpenAI is not None:
            logging.warning("OpenAI API Key is missing. Translation features will be disabled.")
            self.openai_client = None
        elif OpenAI is None:
            self.openai_client = None # Library not installed
        else:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logging.info("OpenAI client initialized.")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}. Translation disabled.")
                self.openai_client = None
        try:
            aai.settings.api_key = self.assemblyai_api_key
            self.aai_client = aai
            logging.info("AssemblyAI SDK configured.")
        except Exception as e:
            logging.critical(f"Failed to configure AssemblyAI SDK: {e}")
            raise ValueError("Failed to configure AssemblyAI SDK.") from e
        # --- End API Initialization ---

        # --- Load YAMNet Model and Classes ---
        self.yamnet_model = None
        self.model_load_type = None # 'hub' or 'local'
        self.yamnet_model, self.model_load_type = self._load_yamnet_model() # Store type
        self.yamnet_classes = self._load_yamnet_classes()
        if not self.yamnet_model or not self.yamnet_classes:
            logging.critical("Failed to initialize YAMNet model or classes. Cannot proceed.")
            raise RuntimeError("Failed to initialize YAMNet model or classes.")
        # --- End Model Loading ---

        # --- Prepare Genre Filters ---
        self.genre_configs = self._prepare_genre_configs()
        # --- End Genre Filters ---

        # --- Initialize Delay Setting ---
        self.non_verbal_delay_ms = 0 # Default to 0ms (no delay)
        # --- End Delay Setting ---

        try: os.makedirs(OUTPUT_DIR, exist_ok=True)
        except OSError as e: logging.error(f"Could not create output directory '{OUTPUT_DIR}': {e}")

        logging.info("SubtitleGenerator initialized successfully.")


    def _load_yamnet_model(self) -> Tuple[Optional[Any], Optional[str]]:
        """Load YAMNet model, preferring TF Hub, falling back to local SavedModel."""
        model = None
        load_type = None
        hub_url = 'https://tfhub.dev/google/yamnet/1'

        # Attempt 1: Load from TensorFlow Hub URL
        logging.info(f"Attempting to load YAMNet model from TF Hub URL: {hub_url}")
        try:
            # Loading from TF Hub typically returns a Keras Layer or compatible object
            model = hub.load(hub_url)
            # Perform dummy inference (TF Hub Keras layer usually callable directly with batch dim)
            # **Note:** The standard Hub model expects a BATCH dimension.
            dummy_input = tf.zeros([1, Config.EXPECTED_SAMPLES], dtype=tf.float32)
            _ = model(dummy_input) # Call directly
            logging.info("Successfully loaded and verified YAMNet from TF Hub.")
            load_type = 'hub'
            return model, load_type
        except ValueError as ve: # Catch specific incompatible type error
            if "incompatible/unknown type" in str(ve) and "tfhub_modules" in str(ve):
                logging.error(f"Failed to load YAMNet from TF Hub: {ve}")
                logging.error("This often indicates a corrupted TF Hub cache. "
                              "Try deleting the cache directory mentioned in the error "
                              "(e.g., C:\\Users\\YourUser\\AppData\\Local\\Temp\\tfhub_modules) "
                              "and run the script again.")
            else: # Re-raise other ValueErrors
                logging.error(f"ValueError loading from TF Hub: {ve}", exc_info=True)
            model = None # Ensure model is None if Hub load fails
        except Exception as e:
            logging.error(f"Unexpected error loading YAMNet model from TF Hub URL '{hub_url}': {e}", exc_info=True)
            model = None # Ensure model is None if Hub load fails

        # Attempt 2: Load from local SavedModel directory if Hub failed and MODEL_DIR is set
        if model is None and MODEL_DIR and os.path.isdir(MODEL_DIR):
            logging.info(f"TF Hub load failed. Attempting to load YAMNet SavedModel from local path: {MODEL_DIR}")
            try:
                model_local = tf.saved_model.load(MODEL_DIR)
                if 'serving_default' not in model_local.signatures:
                    raise ValueError("Loaded local SavedModel missing 'serving_default' signature.")

                # *** FIX for Local Dummy Check ***
                # Check the signature: expects 'waveform' arg, shape (None,) or similar 1D
                # Perform dummy inference matching the expected signature.
                dummy_input_local = tf.zeros([Config.EXPECTED_SAMPLES], dtype=tf.float32) # 1D input
                _ = model_local.signatures['serving_default'](waveform=dummy_input_local) # Use correct arg name

                logging.info("Successfully loaded and verified YAMNet SavedModel from local path.")
                model = model_local # Assign to the main model variable
                load_type = 'local'
                return model, load_type
            except Exception as e:
                logging.error(f"Failed to load YAMNet from local path '{MODEL_DIR}' as well: {e}", exc_info=True)
                return None, None # Both methods failed
        elif model is None:
            logging.error("YAMNet model loading failed. Could not load from TF Hub or local path.")
            return None, None

        # Should not be reached if logic is correct, but return None safety
        return model, load_type


    # (Keep _load_yamnet_classes and _prepare_genre_configs methods as they were correct - unchanged)
    def _load_yamnet_classes(self) -> Optional[List[str]]:
        """Load YAMNet class names, preferring local CSV, falling back to TF Hub model asset."""
        # Attempt 1: Load from local CLASS_MAP_CSV
        local_csv_path = CLASS_MAP_CSV
        if local_csv_path and os.path.exists(local_csv_path):
            logging.info(f"Loading YAMNet class map from local CSV: {local_csv_path}")
            try:
                df = pd.read_csv(local_csv_path)
                if 'display_name' not in df.columns: raise ValueError("CSV missing 'display_name'.")
                class_names = df['display_name'].tolist()
                logging.info(f"Loaded {len(class_names)} class names from local CSV.")
                if len(class_names) != 521: logging.warning(f"Expected 521 classes, found {len(class_names)}.")
                return class_names
            except Exception as e:
                logging.warning(f"Failed to load class names from local CSV '{local_csv_path}': {e}. Trying TF Hub asset...")
        elif local_csv_path:
             logging.warning(f"Local class map CSV '{local_csv_path}' configured but not found. Trying TF Hub asset...")
        else:
             logging.info("Local class map CSV not configured. Trying TF Hub model asset...")


        # Attempt 2: Load from TF Hub model asset (if model loaded successfully from Hub)
        # Check if model exists AND load_type is hub AND it has the attribute
        if self.yamnet_model and self.model_load_type == 'hub' and hasattr(self.yamnet_model, 'class_map_path'):
            try:
                # Ensure the path tensor is executed and decoded
                class_map_path_tensor = self.yamnet_model.class_map_path()
                # Check if it's a tf.Tensor; if so, call .numpy()
                if isinstance(class_map_path_tensor, tf.Tensor):
                    class_map_path_hub = class_map_path_tensor.numpy().decode('utf-8')
                else: # Assume it's already a string-like object if not a Tensor
                    class_map_path_hub = str(class_map_path_tensor)

                logging.info(f"Attempting to load class map from TF Hub model asset: {class_map_path_hub}")
                if os.path.exists(class_map_path_hub):
                    df = pd.read_csv(class_map_path_hub)
                    if 'display_name' not in df.columns: raise ValueError("Hub asset CSV missing 'display_name'.")
                    class_names = df['display_name'].tolist()
                    logging.info(f"Loaded {len(class_names)} class names from TF Hub model asset.")
                    if len(class_names) != 521: logging.warning(f"Expected 521 classes, found {len(class_names)} in Hub asset.")
                    return class_names
                else:
                    logging.error(f"Class map path from TF Hub model does not exist: {class_map_path_hub}")
                    return None
            except Exception as e:
                logging.error(f"Could not load class map from TF Hub asset: {e}")
                return None
        else:
            logging.error("Cannot load class map: No local CSV found/loaded and TF Hub model/asset unavailable.")
            return None

    def _prepare_genre_configs(self) -> Dict[str, Dict[str, Set[str]]]:
        # This function seemed correct, keep as is.
        logging.debug("Preparing genre filter configurations...")
        configs: Dict[str, Dict[str, Set[str]]] = {}
        all_valid_normalized_labels = set(YAMNET_LABEL_MAP.values())

        def get_normalized_set(raw_labels: List[str], genre_name: str, list_type: str) -> Set[str]:
            normalized_set = set()
            for label in raw_labels:
                norm_label = label
                if not (label.startswith('[') and label.endswith(']')):
                    found = False
                    for raw, norm in YAMNET_LABEL_MAP.items():
                        # Case-insensitive comparison for flexibility
                        if label.lower() == raw.lower() or label.lower() == norm.strip('[]').lower():
                            norm_label = norm; found = True; break
                    if not found:
                         # Allow direct use of normalized format (e.g., "[Dog barking]")
                         if label in all_valid_normalized_labels: norm_label = label; found = True
                         # Also check if adding brackets makes it valid (e.g. "Dog barking" -> "[Dog barking]")
                         elif f"[{label}]" in all_valid_normalized_labels: norm_label = f"[{label}]"; found = True

                    if not found:
                        logging.warning(f"Genre '{genre_name}': {list_type.capitalize()} label '{label}' unmatched.")
                        continue
                # Final check if the determined label is actually in our map values
                if norm_label in all_valid_normalized_labels: normalized_set.add(norm_label)
                else:
                    logging.warning(f"Genre '{genre_name}': {list_type.capitalize()} label '{label}' resulted in '{norm_label}' which is not a valid normalized label.")

            return normalized_set

        for genre, filters in GENRE_SOUND_FILTERS.items():
            allowed_raw = filters.get("allow", [])
            blocked_raw = filters.get("block", [])
            allowed_normalized = get_normalized_set(allowed_raw, genre, "allow")
            blocked_normalized = get_normalized_set(blocked_raw, genre, "block")
            # Ensure blocked doesn't remove something explicitly allowed in the same definition
            final_allowed = allowed_normalized - (blocked_normalized & allowed_normalized)
            configs[genre] = {"allowed": final_allowed, "blocked": blocked_normalized}
            logging.debug(f"Genre '{genre}': Normalized Allowed={len(final_allowed)}, Blocked={len(blocked_normalized)}")

        # Prepare General filter (block obscure and specific unwanted sounds)
        general_allowed = set(YAMNET_LABEL_MAP.values())
        obscure_normalized = set()
        for obscure_raw_label in OBSCURE_LABELS:
            norm_val = YAMNET_LABEL_MAP.get(obscure_raw_label)
            if norm_val: obscure_normalized.add(norm_val)
            # Also try finding normalized versions directly
            elif obscure_raw_label in all_valid_normalized_labels: obscure_normalized.add(obscure_raw_label)
            elif f"[{obscure_raw_label}]" in all_valid_normalized_labels: obscure_normalized.add(f"[{obscure_raw_label}]")

        general_blocked = obscure_normalized
        # Explicitly block Silence and Speech from standalone non-verbal in General
        if "[Silence]" in general_allowed: general_blocked.add("[Silence]")
        if "[Speech]" in general_allowed: general_blocked.add("[Speech]")

        configs["General"] = {"allowed": general_allowed - general_blocked, "blocked": general_blocked}
        logging.debug(f"Genre 'General': Allowed={len(configs['General']['allowed'])}, Blocked={len(configs['General']['blocked'])}")
        return configs


    def generate_subtitles(self, audio_path, transcription_language , genre , output_type, delay):
        """Main workflow: Get user prefs, process audio, merge, post-process, save."""
        logging.info("Starting subtitle generation process...")
        if not os.path.exists(audio_path):
            logging.error(f"Processed audio file not found: {audio_path}")
            raise FileNotFoundError(f"Processed audio file not found: {audio_path}")

        # --- Get User Preferences ---
        selected_genres = self._select_genres(genre)
        target_language = self._select_target_language(transcription_language)
        output_format = self._select_output_format(output_type)
        self.non_verbal_delay_ms = self._select_non_verbal_delay(delay) # <-- Get delay preference
        # --- End User Preferences ---

        logging.info(f"Parameters: Genres={selected_genres}, TargetLang={target_language}, "
                     f"OutputFormat={output_format}, NV_Delay={self.non_verbal_delay_ms}ms")

        # --- Core Processing Steps ---
        nonverbal_results = self._process_nonverbal_chunks(selected_genres)
        verbal_results = self._generate_verbal_subtitles(audio_path, transcription_language)
        merged_results = self._merge_subtitles(verbal_results, nonverbal_results)
        final_results = self._post_process_subtitles(merged_results, target_language)
        # --- End Core Processing Steps ---

        # --- Save Outputs ---
        if not selected_genres: model_name_part = "NoFilter"
        elif "General" in selected_genres: model_name_part = "GeneralFilter"
        elif len(selected_genres) == 1: model_name_part = f"{selected_genres[0]}Filter"
        else: genre_abbr = "".join(g[0] for g in sorted(selected_genres))[:10].upper(); model_name_part = f"MultiFilter_{genre_abbr}"

        lang_part = f"_{target_language}" if target_language.lower() != "english" else ""
        delay_part = f"_Delay{self.non_verbal_delay_ms // 1000}s" if self.non_verbal_delay_ms > 0 else ""
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        safe_audio_name = "".join(c if c.isalnum() else "_" for c in audio_basename).strip('_')
        base_filename = f"{safe_audio_name}_{model_name_part}{lang_part}{delay_part}"
        # base_filename = filename

        self._save_outputs(final_results, base_filename, output_format)
        logging.info("Subtitle generation workflow completed.")
        return final_results
    

    def _select_genres(self , selection) -> List[str]:
        """Prompts user to select genre filters for non-verbal sounds."""
        print("\n--- Select Non-Verbal Sound Filters ---")
        print("Choose one or more filters. Non-verbal sounds must pass ALL selected filters.")
        print("Available genre filters:")
        defined_genres = sorted([g for g in self.genre_configs.keys() if g != "General"])
        options = {"G": "General"}
        for i, genre in enumerate(defined_genres): options[str(i+1)] = genre

        print(f"  G. General (Common non-verbal sounds, blocks obscure ones)")
        for num_str, genre_name in options.items():
            if num_str != "G": print(f"  {num_str}. {genre_name}")
        print(f"  A. All Defined Genres (Combines allowed sounds from 1-{len(defined_genres)})")
        print(f"  N. None (Disable non-verbal sound detection)")

        while True:
            print("\nEnter comma-separated numbers (e.g., '1, 3'), 'G', 'A', or 'N':")
            selection = selection.strip().upper()
            if not selection: print("No selection made. Defaulting to 'General'."); logging.info("Defaulted to 'General' genre selection."); return ["General"]
            if selection == 'N': logging.info("User selected 'None'."); print("Non-verbal sound detection disabled."); return []
            if selection == 'G': logging.info("User selected 'General'."); print("Using 'General' filter."); return ["General"]
            if selection == 'A': selected_genre_names = defined_genres; logging.info(f"User selected 'All Defined'. Using genres: {selected_genre_names}"); print(f"Using all defined genre filters: {', '.join(selected_genre_names)}"); return selected_genre_names

            try:
                selected_indices = [item.strip() for item in selection.split(',') if item.strip()]
                temp_selected_names = set()
                valid_selection = True
                for index_str in selected_indices:
                    if index_str.isdigit() and index_str in options: temp_selected_names.add(options[index_str])
                    else: print(f"Invalid selection: '{index_str}'. Use numbers listed, 'G', 'A', or 'N'."); valid_selection = False; break

                if valid_selection and temp_selected_names:
                    final_selection = sorted(list(temp_selected_names))
                    logging.info(f"User selected specific genres: {final_selection}")
                    print(f"Using selected genre filters: {', '.join(final_selection)}")
                    return final_selection
                elif valid_selection: # User entered something, but no valid numbers after split/strip
                    print("No valid genres selected from input.")

            except ValueError: # Should not happen with current checks, but good practice
                print("Invalid input format. Use comma-separated numbers, 'G', 'A', or 'N'.")

    def _select_target_language(self, language) -> str:
        """Prompts user to select a target language for translation."""
        print("\n--- Translation (Optional) ---")

        language = language
        # default_lang = "english"
        # print("Common languages: english, spanish, french, german, japanese, chinese, hindi, etc.")
        # while True:
        #     language = input(f"Enter target language (leave blank for '{default_lang}', type 'none' to skip): ").strip().lower()
        #     if not language: logging.info(f"Using default target language: {default_lang}"); return default_lang
        #     if language == 'none': logging.info("Skipping translation."); return "english" # Return english as it's the base
        if len(language) > 1: # Basic check for a plausible language name
            logging.info(f"Selected target language: {language}")
            if not self.openai_client:
                print(f"Warning: OpenAI client not available. Cannot translate to '{language}'.")
                logging.warning(f"OpenAI client unavailable, cannot translate.")
                return "english" # Fallback to english if translation isn't possible
            return language
        
        # print("Please enter a valid language name (e.g., 'german', 'japanese') or 'none'.")

    def _select_output_format(self , output_type) -> str:
        """Prompts user to select the output subtitle format (SRT or VTT)."""
        print("\n--- Output Format ---")

        default_format = 'srt'

        if output_type is not None:
            default_format = output_type

        return default_format
        
    
        # while True:
        #     choice = input(f"Select output format ('srt' or 'vtt', leave blank for '{default_format}'): ").strip().lower()
        #     if not choice: logging.info(f"Using default output format: {default_format}"); return default_format
        #     if choice in ['srt', 'vtt']: logging.info(f"Selected output format: {choice}"); return choice
        #     print("Invalid choice. Please enter 'srt' or 'vtt'.")

    # --- NEW FUNCTION: Select Non-Verbal Delay ---
    def _select_non_verbal_delay(self , delay) -> int:
        """Prompts user to select minimum delay between non-verbal sounds."""
        # print("\n--- Non-Verbal Sound Delay ---")
        # print("Set a minimum time gap between consecutive non-verbal sounds.")
        # options = {'0': 0, '2': 2, '5': 5, '10': 10, '30': 30}
        # default_delay_sec = 0
        # print("Available options (in seconds):")
        # for key, val in options.items():
        #     print(f"  {key}. {val} seconds" + (" (No delay - Default)" if val == default_delay_sec else ""))

        # while True:
        #     choice = input(f"Enter desired delay number (leave blank for {default_delay_sec}s): ").strip()

        default_delay_sec = 0

        if delay is not None:
            return delay * 1000
    
        return default_delay_sec
        

        # if not choic:
        #     delay_sec = default_delay_sec
        #     logging.info(f"Using default non-verbal delay: {delay_sec}s")
        #     print(f"Using default: {delay_sec} seconds delay.")
        #     return delay_sec * 1000 # Convert to ms

        # if choice in options:
        #     delay_sec = options[choice]
        #     logging.info(f"Selected non-verbal delay: {delay_sec}s")
        #     print(f"Using {delay_sec} seconds delay.")
        #     return delay_sec * 1000 # Convert to ms
            # else:
            #     print(f"Invalid choice. Please enter one of the numbers: {', '.join(options.keys())}.")
    # --- END NEW FUNCTION ---


    def _process_nonverbal_chunks(self, selected_genres: List[str]) -> List[SubtitleEntry]:
        """Process audio chunks with YAMNet using combined genre filters."""
        results: List[SubtitleEntry] = []
        chunk_meta_path = os.path.join(PROCESSING_DIR, "chunk_metadata.json")
        chunk_files_info: List[Dict] = []

        # Load chunk info (prefer metadata file)
        if os.path.exists(chunk_meta_path):
            logging.info(f"Loading chunk information from {chunk_meta_path}")
            try:
                with open(chunk_meta_path, 'r') as f: chunk_files_info = json.load(f)
                if not isinstance(chunk_files_info, list) or (chunk_files_info and not isinstance(chunk_files_info[0], dict)):
                    logging.warning(f"Chunk metadata {chunk_meta_path} bad format. Scanning dir."); chunk_files_info = []
                # Add absolute path if only relative paths stored
                if chunk_files_info and 'path' in chunk_files_info[0] and not os.path.isabs(chunk_files_info[0]['path']):
                     for info in chunk_files_info: info['path'] = os.path.join(PROCESSING_DIR, info['path'])
            except Exception as e:
                logging.error(f"Error reading chunk metadata {chunk_meta_path}: {e}. Scanning dir.", exc_info=True); chunk_files_info = []

        if not chunk_files_info:
            logging.info("Chunk metadata not found/invalid. Scanning dir for chunk_*.wav.")
            found_files = []
            try: # Defensive check for directory existence
                found_files = sorted([f for f in os.listdir(PROCESSING_DIR) if f.startswith("chunk_") and f.endswith(".wav")])
            except FileNotFoundError:
                 logging.error(f"Processing directory '{PROCESSING_DIR}' does not exist for chunk scan.")
                 return []

            for chunk_file in found_files:
                try:
                    parts = chunk_file.replace("chunk_", "").replace(".wav", "").split("_")
                    # Ensure enough parts exist before accessing indices
                    if len(parts) >= 3:
                        chunk_files_info.append({
                            "index": int(parts[0]),
                            "start": int(parts[1]),
                            "end": int(parts[2]),
                            "path": os.path.join(PROCESSING_DIR, chunk_file),
                            # Estimate duration, actual chunk length might vary slightly
                            "duration": int(Config.CHUNK_DURATION_SEC * 1000)
                         })
                    else:
                        logging.warning(f"Could not parse timing/index from fallback filename scan (not enough parts): {chunk_file}")
                except (IndexError, ValueError):
                    logging.warning(f"Could not parse timing/index from fallback filename scan (error during conversion): {chunk_file}")

        if not chunk_files_info: logging.warning("No audio chunks found. Cannot perform non-verbal detection."); return []
        if not selected_genres: logging.info("No genres selected. Skipping non-verbal detection."); return []

        logging.info(f"Processing {len(chunk_files_info)} chunks for non-verbal sounds (Genres: {selected_genres})...")

        # --- Determine combined allowed/blocked classes (same logic as before) ---
        combined_allowed_labels: Set[str] = set()
        temp_allowed_union = set()
        # Apply genre-specific allow lists first
        genres_to_use = selected_genres
        if "General" in selected_genres:
             genres_to_use = ["General"] # General overrides specific if both selected
             logging.info("Using 'General' filter (overrides specific selections if present).")
        elif not genres_to_use: # Handle case where only "None" might have been selected (empty list)
            logging.warning("No valid genres selected for filtering. Non-verbal detection will yield no results.")
            return []

        # Union of allowed labels from selected genres
        for genre in genres_to_use:
            if genre in self.genre_configs:
                temp_allowed_union.update(self.genre_configs[genre]["allowed"])
            else:
                logging.warning(f"Selected genre '{genre}' not found in configurations.")

        # Union of blocked labels from selected genres
        temp_blocked_union = set()
        for genre in genres_to_use:
            if genre in self.genre_configs:
                temp_blocked_union.update(self.genre_configs[genre]["blocked"])

        # Final allowed set is allowed MINUS blocked
        combined_allowed_labels = temp_allowed_union - temp_blocked_union
        logging.info(f"Combined Filter: {len(combined_allowed_labels)} allowed labels after applying blocks.")
        if not combined_allowed_labels: logging.warning("No labels allowed by filters. Non-verbal detection will yield no results."); return []
        # --- End Filter Combination ---

        # --- YAMNet Inference Loop ---
        for chunk_info in tqdm(chunk_files_info, desc="Detecting Non-Verbal Sounds"):
            chunk_path = chunk_info['path']
            start_time_ms = chunk_info['start']
            end_time_ms = chunk_info['end']
            if not os.path.exists(chunk_path): logging.warning(f"Chunk file not found: {chunk_path}"); continue

            waveform = load_and_prepare_chunk(chunk_path, Config.TARGET_SR, Config.EXPECTED_SAMPLES)
            if waveform is None: logging.warning(f"Skipping chunk {os.path.basename(chunk_path)} (loading error)."); continue

            # *** ADAPT INFERENCE CALL BASED ON LOAD TYPE ***
            try:
                if self.model_load_type == 'hub':
                    # Hub model expects BATCHED input
                    waveform_tensor = tf.constant(waveform[np.newaxis, :], dtype=tf.float32)
                    scores, _, _ = self.yamnet_model(waveform_tensor) # Direct call
                    scores_np = scores.numpy().squeeze() # Squeeze batch dim
                elif self.model_load_type == 'local':
                    # Local model signature expects 1D input with arg name 'waveform'
                    waveform_tensor = tf.constant(waveform, dtype=tf.float32) # 1D tensor
                    prediction = self.yamnet_model.signatures['serving_default'](waveform=waveform_tensor)
                    # Find scores tensor (key might be 'scores', 'output_0' etc.)
                    scores_key = 'output_0' # Assume based on common TF practice
                    # Try to find a better key if 'output_0' doesn't work or exists but isn't right shape
                    possible_keys = ['scores', 'prediction', 'probabilities']
                    if scores_key not in prediction:
                        found_key = None
                        for key in possible_keys:
                             if key in prediction: found_key = key; break
                        if not found_key:
                            # If common keys aren't present, fall back to first key if only one output
                            if len(prediction) == 1: found_key = list(prediction.keys())[0]
                            else: raise KeyError(f"Could not find scores tensor in local model output. Keys: {prediction.keys()}")
                        scores_key = found_key
                        logging.debug(f"Using '{scores_key}' as scores tensor key for local model.")

                    scores = prediction[scores_key]
                    scores_np = scores.numpy().squeeze() # Output is likely already (1, N), squeeze batch
                    # If output was e.g., (N_frames, N_classes), average over frames
                    if scores_np.ndim > 1 and scores_np.shape[0] > 1:
                        logging.warning(f"Local model output scores have shape {scores_np.shape}. Taking mean over axis 0.")
                        scores_np = np.mean(scores_np, axis=0)
                    # Ensure final shape is 1D
                    scores_np = scores_np.flatten()

                else: # Should not happen
                    logging.error("Unknown model load type. Cannot perform inference.")
                    continue

            except Exception as e:
                logging.error(f"YAMNet inference failed for chunk {os.path.basename(chunk_path)}: {e}", exc_info=True)
                continue
            # *** END ADAPTED INFERENCE ***

            # --- Process Scores and Apply Filters (same logic as before) ---
            sorted_indices = np.argsort(scores_np)[::-1]
            best_score_for_chunk = -1.0
            best_label_for_chunk = None

            for idx in sorted_indices:
                confidence = float(scores_np[idx])
                # Stop if confidence drops below threshold
                if confidence < Config.NONVERBAL_CONFIDENCE_THRESHOLD: break

                try: raw_label = self.yamnet_classes[idx]
                except IndexError: logging.warning(f"Index {idx} out of bounds for yamnet_classes."); continue

                normalized_label = YAMNET_LABEL_MAP.get(raw_label)
                # Skip if not mapped or not in the allowed list for the selected genres
                if not normalized_label or normalized_label not in combined_allowed_labels: continue

                # Found the highest confidence allowed sound for this chunk
                best_score_for_chunk = confidence
                best_label_for_chunk = normalized_label
                break # Take only the top valid prediction per chunk

            if best_label_for_chunk:
                 results.append(SubtitleEntry(start_time=start_time_ms, end_time=end_time_ms, text=best_label_for_chunk, confidence=best_score_for_chunk, type="non_verbal"))
                 logging.debug(f"Chunk {chunk_info.get('index', '?')}: Detected '{best_label_for_chunk}' (Conf: {best_score_for_chunk:.2f})")
            # --- End Score Processing ---

        # --- Consolidate consecutive identical non-verbal labels ---
        if not results: logging.info("No raw non-verbal events detected after filtering."); return []
        consolidated: List[SubtitleEntry] = []
        results.sort(key=lambda r: r.start_time) # Ensure sorted by time

        current_sub = results[0]
        for next_sub in results[1:]:
            # Merge if same label and gap is small enough (use a generous gap for consolidation)
            max_merge_gap_ms = Config.MIN_GAP_MS * 2 # Allow slightly larger gap for consolidation than final display
            time_gap = next_sub.start_time - current_sub.end_time

            if next_sub.text == current_sub.text and time_gap < max_merge_gap_ms:
                # Extend the end time and keep the higher confidence
                current_sub.end_time = max(current_sub.end_time, next_sub.end_time)
                current_sub.confidence = max(current_sub.confidence, next_sub.confidence)
                # Don't update start time, keep the earliest start
            else:
                # Finish the current consolidated sub and start a new one
                consolidated.append(current_sub)
                current_sub = next_sub

        consolidated.append(current_sub) # Add the last consolidated subtitle
        logging.info(f"Detected {len(results)} raw non-verbal events, consolidated into {len(consolidated)} events based on proximity.")
        return consolidated


    # (Keep _generate_verbal_subtitles, _merge_subtitles - they seemed okay - unchanged)
    def _generate_verbal_subtitles(self, audio_path: str, language: str) -> List[SubtitleEntry]:
        """Generates verbal subtitles using AssemblyAI."""
        logging.info(f"Starting AssemblyAI transcription for {os.path.basename(audio_path)} (Language: {language})...")
        results: List[SubtitleEntry] = []
        if not self.assemblyai_api_key: logging.error("AssemblyAI API key not configured."); return []

        for attempt in range(Config.ASSEMBLYAI_RETRY_ATTEMPTS):
            try:
                transcriber = self.aai_client.Transcriber()
                # Simple language code mapping (expand if needed)
                lang_code = language
                if language == 'english': lang_code = 'en'
                elif language == 'spanish': lang_code = 'es'
                elif language == 'french': lang_code = 'fr'
                elif language == 'german': lang_code = 'de'
                # Use None if language is not a specific code AssemblyAI recognizes or if 'auto' preferred
                config_lang_code = lang_code if lang_code in ['en', 'es', 'fr', 'de'] else None # Add more valid codes

                config = self.aai_client.TranscriptionConfig(
                    language_code=config_lang_code,
                    punctuate=True,
                    format_text=True, # Enables features like number formatting
                    disfluencies=False # Remove filler words like "um", "uh"
                )
                logging.info(f"Submitting transcription job (Attempt {attempt + 1}) with lang_code='{config_lang_code}'...")

                transcript = transcriber.transcribe(audio_path, config=config)

                if transcript.status == self.aai_client.TranscriptStatus.error:
                    logging.error(f"AAI transcription failed attempt {attempt + 1}: {transcript.error}")
                    # Raise specific errors for retrying vs permanent failure if possible
                    raise ConnectionError(f"AAI Error: {transcript.error}")

                if transcript.status != self.aai_client.TranscriptStatus.completed:
                    logging.warning(f"AAI status is {transcript.status}. May need more time or failed.")
                    # If queued or processing, could wait, but for simplicity, we retry/fail here
                    raise TimeoutError(f"AAI Status is {transcript.status}, expected 'completed'.")

                logging.info("AAI transcription completed successfully.")
                if not transcript.words:
                    logging.warning("AAI transcription successful but returned no words.")
                    return []

                # Group words into sentences/phrases for subtitles
                # More robust sentence splitting logic could be added here if needed
                current_sentence_words = []
                current_start_ms = None
                last_word_end_ms = 0

                for i, word in enumerate(transcript.words):
                    if word.start is None or word.end is None: continue # Skip words without timing

                    if current_start_ms is None:
                        current_start_ms = word.start

                    current_sentence_words.append(word)
                    last_word_end_ms = word.end

                    # Determine if this word ends a subtitle segment
                    is_end_of_segment = False
                    ends_with_punctuation = word.text.strip().endswith(('.', '!', '?'))

                    # Check gap to next word
                    gap_to_next_ms = float('inf')
                    if i < len(transcript.words) - 1 and transcript.words[i+1].start is not None:
                        gap_to_next_ms = transcript.words[i+1].start - word.end

                    # Define segment break conditions
                    max_segment_duration_ms = Config.MAX_SUBTITLE_DURATION_MS - 500 # Slightly less for buffer
                    segment_duration = last_word_end_ms - current_start_ms
                    max_words_per_sub = 15 # Arbitrary limit

                    if ends_with_punctuation and segment_duration > 500: # Min duration for punctuation break
                        is_end_of_segment = True
                    elif gap_to_next_ms > 700: # Significant pause
                        is_end_of_segment = True
                    elif segment_duration > max_segment_duration_ms:
                        is_end_of_segment = True
                    elif len(current_sentence_words) > max_words_per_sub:
                         is_end_of_segment = True
                    elif i == len(transcript.words) - 1: # Last word
                        is_end_of_segment = True

                    if is_end_of_segment and current_sentence_words:
                        sentence_text = " ".join([w.text for w in current_sentence_words]).strip()
                        if sentence_text: # Ensure text is not empty
                            confidences = [w.confidence for w in current_sentence_words if w.confidence is not None]
                            avg_conf = float(np.mean(confidences)) if confidences else 0.0

                            # Final validation of times
                            if current_start_ms is not None and last_word_end_ms is not None and current_start_ms < last_word_end_ms:
                                results.append(SubtitleEntry(
                                    start_time=current_start_ms,
                                    end_time=last_word_end_ms,
                                    text=sentence_text,
                                    confidence=avg_conf,
                                    type="verbal"
                                ))
                            else:
                                logging.warning(f"Skipping verbal entry due to invalid times: {current_start_ms} -> {last_word_end_ms} for text '{sentence_text}'")

                        # Reset for next segment
                        current_sentence_words = []
                        current_start_ms = None

                logging.info(f"Processed {len(transcript.words)} AAI words into {len(results)} verbal subtitle entries.")
                return results # Success

            except (ConnectionError, TimeoutError, RuntimeError, Exception) as e:
                logging.warning(f"AAI attempt {attempt + 1}/{Config.ASSEMBLYAI_RETRY_ATTEMPTS} failed: {str(e)}")
                if attempt == Config.ASSEMBLYAI_RETRY_ATTEMPTS - 1:
                    logging.error("AAI transcription failed after all retries.", exc_info=True)
                    return [] # Failed all attempts
                logging.info(f"Retrying in {Config.ASSEMBLYAI_RETRY_DELAY_SEC} seconds...")
                time.sleep(Config.ASSEMBLYAI_RETRY_DELAY_SEC)

        return [] # Should not be reached if logic is correct

    def _merge_subtitles(self, verbal: List[SubtitleEntry], nonverbal: List[SubtitleEntry]) -> List[SubtitleEntry]:
        """Merges non-verbal sounds into nearby verbal subtitles or keeps them standalone."""
        logging.info(f"Merging {len(verbal)} verbal and {len(nonverbal)} non-verbal entries...")
        merged: List[SubtitleEntry] = []
        nv_used_indices: Set[int] = set() # Track non-verbal subs already merged

        # Ensure lists are sorted by start time
        verbal.sort(key=lambda x: x.start_time)
        nonverbal.sort(key=lambda x: x.start_time)

        # Iterate through verbal subtitles to find potential merges
        for v_idx, v_sub in enumerate(verbal):
            best_nv_to_merge: Optional[SubtitleEntry] = None
            best_nv_idx = -1
            best_nv_overlap = -1 # Measure proximity/overlap

            # Find the best overlapping/nearby non-verbal sound
            for nv_idx, nv_sub in enumerate(nonverbal):
                if nv_idx in nv_used_indices: continue # Skip if already used

                # Calculate overlap duration
                overlap_start = max(v_sub.start_time, nv_sub.start_time)
                overlap_end = min(v_sub.end_time, nv_sub.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Calculate proximity (time between centers)
                v_center = v_sub.start_time + (v_sub.end_time - v_sub.start_time) / 2
                nv_center = nv_sub.start_time + (nv_sub.end_time - nv_sub.start_time) / 2
                proximity = abs(v_center - nv_center)

                # Define merge conditions (prioritize overlap, then proximity)
                # Allow merge if overlap is significant OR centers are very close
                should_consider_merge = False
                min_overlap_for_merge = 100 # ms
                max_proximity_for_merge = Config.MAX_OVERLAP_MS + 200 # Generous proximity

                if overlap_duration >= min_overlap_for_merge:
                     should_consider_merge = True
                elif proximity < max_proximity_for_merge:
                     should_consider_merge = True

                if should_consider_merge:
                    # Prefer higher confidence NV sound if multiple candidates
                    # Or prioritize the one with more overlap if confidences are similar
                    current_score = overlap_duration + nv_sub.confidence * 1000 # Simple scoring
                    if best_nv_to_merge is None or current_score > best_nv_overlap:
                            best_nv_to_merge = nv_sub
                            best_nv_idx = nv_idx
                            best_nv_overlap = current_score


            # If a suitable non-verbal sound was found, merge it
            if best_nv_to_merge:
                # Simple append: Verbal text + [NonVerbal Sound]
                # Ensure NV tag is distinct
                merged_text = f"{v_sub.text} {best_nv_to_merge.text}"

                # Adjust timing to encompass both
                merged_start = min(v_sub.start_time, best_nv_to_merge.start_time)
                merged_end = max(v_sub.end_time, best_nv_to_merge.end_time)

                # Average confidence (or other weighting)
                merged_conf = (v_sub.confidence + best_nv_to_merge.confidence) / 2

                merged.append(SubtitleEntry(
                    start_time=merged_start,
                    end_time=merged_end,
                    text=merged_text,
                    type="combined",
                    confidence=merged_conf
                ))
                nv_used_indices.add(best_nv_idx) # Mark this NV as used
                logging.debug(f"Merged V '{v_sub.text[:20]}...' with NV '{best_nv_to_merge.text}'")
            else:
                # No suitable non-verbal found, add the verbal sub as is
                merged.append(v_sub)

        # Add remaining standalone non-verbal sounds (if confidence is high enough)
        num_standalone = 0
        for nv_idx, nv_sub in enumerate(nonverbal):
            if nv_idx not in nv_used_indices:
                if nv_sub.confidence >= Config.STANDALONE_CONFIDENCE_THRESHOLD:
                    merged.append(nv_sub)
                    num_standalone += 1
                else:
                    logging.debug(f"Skipping standalone NV '{nv_sub.text}' (Conf {nv_sub.confidence:.2f} < {Config.STANDALONE_CONFIDENCE_THRESHOLD})")

        logging.info(f"Merging done. Added {num_standalone} standalone non-verbal events.")

        # Final sort of the merged list by start time
        merged.sort(key=lambda x: x.start_time)
        logging.info(f"Merged list size after adding standalone NV: {len(merged)}.")
        return merged


    # --- MODIFIED FUNCTION: _post_process_subtitles ---
    # --- REFINED FUNCTION: _post_process_subtitles ---
    def _post_process_subtitles(self, subtitles: List[SubtitleEntry], target_language: str) -> List[SubtitleEntry]:
        """Adjusts timing, applies NV-NV delay, translates, and formats subtitles."""
        if not subtitles: return []
        logging.info(f"Post-processing {len(subtitles)} entries (Target: {target_language}, NV-NV Delay: {self.non_verbal_delay_ms}ms).")

        adjusted_subs: List[SubtitleEntry] = []
        last_sub_end_time = -float('inf') # Tracks end time of the *previous* subtitle added
        previous_sub_type = None          # Tracks type of the *previous* subtitle added

        for i, sub in enumerate(subtitles):
            # Get the original start time for potential delay calculation
            current_start_time = sub.start_time

            # --- Apply Non-Verbal to Non-Verbal Delay ---
            # Check if delay is enabled, current is NV, and previous was also NV
            if sub.type == 'non_verbal' and previous_sub_type == 'non_verbal' and self.non_verbal_delay_ms > 0:
                required_start_time = last_sub_end_time + self.non_verbal_delay_ms
                if current_start_time < required_start_time:
                    adjusted_start_for_delay = required_start_time
                    logging.debug(f"Sub {i} ('{sub.text}'): Applying NV-NV delay. Prev NV ended at {last_sub_end_time}, "
                                  f"Original start {sub.start_time}, Required >= {required_start_time}. New effective start: {adjusted_start_for_delay}")
                    # Use the delayed start time for subsequent calculations in this iteration
                    current_start_time = adjusted_start_for_delay
                # else: # Optional log if condition met but delay wasn't needed
                #    logging.debug(f"Sub {i} ('{sub.text}'): NV-NV condition met, but original start {current_start_time} >= required {required_start_time}. No delay needed.")

            # --- End Non-Verbal to Non-Verbal Delay ---


            # --- Standard Timing Adjustments (Min Gap, Duration Limits) ---
            # Ensure start time is non-negative and respects the minimum gap from the *previous* subtitle's end
            start = max(0, current_start_time)
            start = max(start, last_sub_end_time + Config.MIN_GAP_MS)

            # Calculate original duration based on the subtitle's original times
            original_duration = max(0, sub.end_time - sub.start_time)
            # Apply min/max duration constraints
            duration = max(Config.MIN_SUBTITLE_DURATION_MS, original_duration)
            duration = min(Config.MAX_SUBTITLE_DURATION_MS, duration)

            # Calculate final end time based on adjusted start and constrained duration
            end = start + duration

            # Sanity check: Ensure start < end after all adjustments
            if start >= end:
                logging.warning(f"Sub {i} ('{sub.text}') has invalid timing after all adjustments: Start={start}, End={end}. Skipping.")
                # Crucially, DO NOT update last_sub_end_time or previous_sub_type if skipping
                continue
            # --- End Standard Timing Adjustments ---

            # Create a new SubtitleEntry with adjusted times but original text/type/confidence for now
            # Text will be handled during translation and formatting steps
            adjusted_sub = dc_replace(sub, start_time=start, end_time=end)
            adjusted_subs.append(adjusted_sub)

            # --- Update tracking variables for the NEXT iteration ---
            last_sub_end_time = end           # Use the final calculated end time of the sub just added
            previous_sub_type = sub.type      # Record the type of the sub just added

        logging.info(f"Timing adjusted for {len(adjusted_subs)} subtitles (incl. NV-NV delay).")

        # --- Translation ---
        if target_language.lower() != "english" and self.openai_client:
            logging.info(f"Translating {len(adjusted_subs)} subtitles to {target_language}...")
            translated_subs = self._translate_subtitles_openai(adjusted_subs, target_language)
            logging.info(f"Translation result count: {len(translated_subs)}")
        else:
            if target_language.lower() != "english":
                logging.warning(f"Skipping translation (OpenAI client unavailable or target is English).")
            translated_subs = adjusted_subs # No translation needed/possible
        # --- End Translation ---

        # --- Text Formatting (Line Breaks, Capitalization) ---
        formatted_subs: List[SubtitleEntry] = []
        for sub in translated_subs:
            text = sub.text.strip()
            if not text:
                logging.debug(f"Skipping empty subtitle at time {sub.start_time}.")
                continue

            # Basic capitalization (optional, could be refined)
            if text and text[0].islower() and not text.startswith('['):
                text = text[0].upper() + text[1:]

            # Split text into lines respecting max width/lines
            lines = self._split_subtitle_text(text)
            formatted_text = "\n".join(lines)

            formatted_subs.append(dc_replace(sub, text=formatted_text))
        # --- End Text Formatting ---

        logging.info(f"Final formatted subtitle count: {len(formatted_subs)}")
        return formatted_subs
    # --- END MODIFIED FUNCTION ---

    # (Keep _split_subtitle_text, _translate_subtitles_openai, _save_outputs methods - they seemed okay - unchanged)
    def _split_subtitle_text(self, text: str) -> List[str]:
        """Splits text into lines based on MAX_CHARS_PER_LINE and MAX_LINES_PER_SUBTITLE."""
        # Handle pre-existing newlines (e.g., from combined subs)
        raw_lines = text.split('\n')
        processed_lines = []

        for line in raw_lines:
            words = line.split()
            if not words: continue # Skip empty lines resulting from split

            current_line = ""
            for word in words:
                # Test adding the next word
                test_line = current_line + (" " if current_line else "") + word

                if len(test_line) <= Config.MAX_CHARS_PER_LINE:
                    # Word fits, add it to the current line
                    current_line = test_line
                else:
                    # Word doesn't fit. Finalize the current line (if not empty)
                    if current_line:
                        processed_lines.append(current_line)
                    # Start a new line with the current word
                    current_line = word
                    # Handle case where a single word is too long
                    if len(current_line) > Config.MAX_CHARS_PER_LINE:
                        logging.warning(f"Word '{current_line}' exceeds MAX_CHARS_PER_LINE ({Config.MAX_CHARS_PER_LINE}). Keeping it on its own line.")
                        processed_lines.append(current_line)
                        current_line = "" # Reset line as the long word is now added

            # Add the last constructed line if it's not empty
            if current_line:
                processed_lines.append(current_line)

        # Enforce maximum number of lines per subtitle
        if len(processed_lines) > Config.MAX_LINES_PER_SUBTITLE:
            logging.warning(f"Subtitle generated {len(processed_lines)} lines, truncating to {Config.MAX_LINES_PER_SUBTITLE}. Original text: '{text}'")
            final_lines = processed_lines[:Config.MAX_LINES_PER_SUBTITLE]
        else:
            final_lines = processed_lines

        return final_lines

    def _translate_subtitles_openai(self, subtitles: List[SubtitleEntry], target_language: str) -> List[SubtitleEntry]:
        """Translates subtitle text using OpenAI, skipping tags."""
        if not self.openai_client:
            logging.error("OpenAI client not available for translation.")
            return subtitles # Return original if client missing

        translated_results: List[SubtitleEntry] = []
        total_subs = len(subtitles)

        for i, sub in enumerate(tqdm(subtitles, desc=f"Translating to {target_language}")):
            text_to_translate = sub.text
            translated_text = text_to_translate # Default to original text

            # Identify simple tags like "[Sound]" to skip translation
            # More complex logic might be needed if tags appear mid-text
            is_simple_tag = text_to_translate.startswith('[') and text_to_translate.endswith(']') and ' ' not in text_to_translate.strip('[]')

            if is_simple_tag:
                logging.debug(f"Skipping translation for tag: {text_to_translate}")
                translated_results.append(sub) # Keep original tag
                continue

            # Attempt translation for non-tags
            for attempt in range(Config.OPENAI_RETRY_ATTEMPTS):
                try:
                    # Construct a clear prompt
                    prompt = (
                        f"Translate the following English subtitle text accurately and concisely into {target_language}. "
                        f"If the text contains bracketed tags (like '[Sound]' or '[Music playing]'), preserve those tags exactly as they are, translating only the text around them. "
                        f"Keep the translation natural and suitable for subtitles.\n\n"
                        f"Original Text: \"{text_to_translate}\"\n\n"
                        f"Translated Text ({target_language}):"
                    )

                    response = self.openai_client.chat.completions.create(
                        model=Config.OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=Config.OPENAI_MAX_TOKENS_PER_SUB * 2, # Allow more tokens for translation
                        temperature=0.3, # Lower temperature for more deterministic translation
                        n=1,
                        stop=None # Let the model decide when to stop
                    )

                    translation = response.choices[0].message.content.strip().strip('"')

                    # Basic validation of the translation
                    if not translation or translation.lower() == text_to_translate.lower():
                        logging.warning(f"Translation for '{text_to_translate[:30]}...' was empty or unchanged.")
                        translated_text = text_to_translate # Keep original if translation failed
                    else:
                        translated_text = translation
                        logging.debug(f"Translated '{text_to_translate[:30]}...' -> '{translated_text[:30]}...'")
                    break # Success, exit retry loop

                except Exception as e:
                    logging.warning(f"OpenAI translation attempt {attempt + 1}/{Config.OPENAI_RETRY_ATTEMPTS} failed for '{text_to_translate[:30]}...': {str(e)}")
                    if attempt == Config.OPENAI_RETRY_ATTEMPTS - 1:
                        logging.error(f"Translation failed permanently for '{text_to_translate[:30]}...'. Using original text.", exc_info=False) # Don't need full stack trace always
                        translated_text = text_to_translate # Use original on final failure
                    else:
                        time.sleep(Config.OPENAI_RETRY_DELAY_SEC) # Wait before retrying

            # Append the (potentially) translated subtitle
            translated_results.append(dc_replace(sub, text=translated_text))

        return translated_results

    def _save_outputs(self, results: List[SubtitleEntry], base_filename: str, output_format: str):
        """Saves the final list of subtitles to a file in SRT or VTT format."""
        if not results:
            logging.warning("No subtitle entries generated to save.")
            print("\nWarning: No subtitles were generated. Nothing to save.")
            return

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        except OSError as e:
            logging.error(f"Cannot save subtitles: Failed to create output directory '{OUTPUT_DIR}': {e}")
            print(f"\nError: Could not create output directory. Subtitles not saved.")
            return

        output_path = os.path.join(OUTPUT_DIR, f"{base_filename}.{output_format.lower()}")
        logging.info(f"Saving {len(results)} final subtitle entries to: {output_path} (Format: {output_format.upper()})")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if output_format == 'vtt':
                    f.write("WEBVTT\n\n") # VTT header

                for i, sub in enumerate(results, 1):
                    # Format timestamps according to the selected format
                    start_ts = format_timestamp(sub.start_time, output_format)
                    end_ts = format_timestamp(sub.end_time, output_format)
                    text = sub.text.strip() # Ensure no leading/trailing whitespace

                    # Final sanity checks before writing
                    if sub.start_time >= sub.end_time:
                        logging.warning(f"Skipping subtitle index {i} during save due to invalid timing: {sub.start_time} >= {sub.end_time}")
                        continue
                    if not text:
                        logging.warning(f"Skipping subtitle index {i} during save due to empty text.")
                        continue

                    # Write the entry
                    f.write(f"{i}\n") # Sequence number
                    f.write(f"{start_ts} --> {end_ts}\n") # Timestamps
                    f.write(f"{text}\n\n") # Subtitle text (potentially multi-line)

            print(f"\nSuccessfully saved subtitles to: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save subtitles to {output_path}: {e}", exc_info=True)
            print(f"\nError: Failed to save subtitles file.")


def main(transcription_language, genre, output_type, delay):

    result = {
        'success': False,
        'data': None,
        'error': None,
        'warnings': []
    }

    print("=" * 60)
    print(" Advanced Subtitle Generator (YAMNet + AssemblyAI + OpenAI)")
    print("=" * 60)
    print("Initializing...")
    print("  Ensure required libraries are installed:")
    print("    pip install tensorflow tensorflow_hub assemblyai openai pandas numpy tqdm librosa")
    print("  (Optional but Recommended for audio processing: pip install pydub)")
    print("  Ensure FFmpeg is installed and accessible in your system PATH if using pydub features.")
    print("  Recommendation: Delete TF Hub cache if encountering download issues:")
    print("-" * 60)

    # Locate Processed Audio (using the exact logic from the original script)
    processed_audio_file = None
    if os.path.isdir(PROCESSING_DIR):
        try:
            processed_files = sorted(
                [os.path.join(PROCESSING_DIR, f) for f in os.listdir(PROCESSING_DIR) if f.startswith("processed_") and f.endswith(".wav")],
                key=os.path.getmtime, reverse=True
            )
            if processed_files:
                processed_audio_file = processed_files[0]
                logging.info(f"Found most recent processed audio: {processed_audio_file}")
            else:
                logging.error(f"No 'processed_*.wav' file found in {PROCESSING_DIR}")
        except Exception as e:
            logging.error(f"Error scanning for processed files in '{PROCESSING_DIR}': {e}")
    else:
        logging.error(f"Processing directory '{PROCESSING_DIR}' not found.")

    if not processed_audio_file:
        print(f"\nError: Could not find a 'processed_*.wav' audio file in the specified processing directory:")
        print(f"'{PROCESSING_DIR}'")
        print("Please ensure you have run the pre-processing script (e.g., Yam_Proc.py) first,")
        print("and that it successfully created the processed audio file in that location.")
        exit(1)

    print(f"\nUsing audio file: {os.path.basename(processed_audio_file)}")

    # Run Generation
    try:
        generator = SubtitleGenerator()
        # Assuming English transcription for now, could be made dynamic
        final_subtitles = generator.generate_subtitles(processed_audio_file, transcription_language , genre, output_type, delay)

        # Display Summary
        print(f"\n--- Generation Summary ---")
        if final_subtitles:
            print(f"Generated {len(final_subtitles)} final subtitle entries.")
            print("First 10 entries (approx):")
            for i, sub in enumerate(final_subtitles[:10]):
                start_f = format_timestamp(sub.start_time)
                end_f = format_timestamp(sub.end_time)
                # Replace newline with ' / ' for compact preview
                text_preview = sub.text.replace('\n', ' / ')
                print(f"  {i+1}. {start_f} --> {end_f}")
                print(f"     {text_preview}") # Indent text line
            if len(final_subtitles) > 10:
                print(f"  ... and {len(final_subtitles) - 10} more entries.")
        else:
            print("No subtitles were generated or saved.")
        print("-" * 26)
        print(f"Log file generated at: {LOG_FILE}")
        print(f"Output files will be saved in: {OUTPUT_DIR}")

        result.update({
            'success': True
        })

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\nError during initialization or processing: {str(e)}")
        result['error'] = str(e)
        logging.critical(f"Script failed due to configuration or file error: {str(e)}", exc_info=True)
    

    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        result['error'] = str(e)
        logging.critical("Script failed unexpectedly during execution.", exc_info=True)
    finally:
        print("\nSubtitle Generation Script Finished.")

    return result



# --- Main Execution Block ---
if __name__ == "__main__":
    print("=" * 60)
    print(" Advanced Subtitle Generator (YAMNet + AssemblyAI + OpenAI)")
    print("=" * 60)
    print("Initializing...")
    print("  Ensure required libraries are installed:")
    print("    pip install tensorflow tensorflow_hub assemblyai openai pandas numpy tqdm librosa")
    print("  (Optional but Recommended for audio processing: pip install pydub)")
    print("  Ensure FFmpeg is installed and accessible in your system PATH if using pydub features.")
    print("  Recommendation: Delete TF Hub cache if encountering download issues:")
    print("-" * 60)

    # Locate Processed Audio (using the exact logic from the original script)
    processed_audio_file = None
    if os.path.isdir(PROCESSING_DIR):
        try:
            processed_files = sorted(
                [os.path.join(PROCESSING_DIR, f) for f in os.listdir(PROCESSING_DIR) if f.startswith("processed_") and f.endswith(".wav")],
                key=os.path.getmtime, reverse=True
            )
            if processed_files:
                processed_audio_file = processed_files[0]
                logging.info(f"Found most recent processed audio: {processed_audio_file}")
            else:
                logging.error(f"No 'processed_*.wav' file found in {PROCESSING_DIR}")
        except Exception as e:
            logging.error(f"Error scanning for processed files in '{PROCESSING_DIR}': {e}")
    else:
        logging.error(f"Processing directory '{PROCESSING_DIR}' not found.")

    if not processed_audio_file:
        print(f"\nError: Could not find a 'processed_*.wav' audio file in the specified processing directory:")
        print(f"'{PROCESSING_DIR}'")
        print("Please ensure you have run the pre-processing script (e.g., Yam_Proc.py) first,")
        print("and that it successfully created the processed audio file in that location.")
        exit(1)

    print(f"\nUsing audio file: {os.path.basename(processed_audio_file)}")

    # Run Generation
    try:
        generator = SubtitleGenerator()
        # Assuming English transcription for now, could be made dynamic
        final_subtitles = generator.generate_subtitles(audio_path=processed_audio_file, transcription_language="en")

        # Display Summary
        print(f"\n--- Generation Summary ---")
        if final_subtitles:
            print(f"Generated {len(final_subtitles)} final subtitle entries.")
            print("First 10 entries (approx):")
            for i, sub in enumerate(final_subtitles[:10]):
                start_f = format_timestamp(sub.start_time)
                end_f = format_timestamp(sub.end_time)
                # Replace newline with ' / ' for compact preview
                text_preview = sub.text.replace('\n', ' / ')
                print(f"  {i+1}. {start_f} --> {end_f}")
                print(f"     {text_preview}") # Indent text line
            if len(final_subtitles) > 10:
                print(f"  ... and {len(final_subtitles) - 10} more entries.")
        else:
            print("No subtitles were generated or saved.")
        print("-" * 26)
        print(f"Log file generated at: {LOG_FILE}")
        print(f"Output files will be saved in: {OUTPUT_DIR}")

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\nError during initialization or processing: {str(e)}")
        logging.critical(f"Script failed due to configuration or file error: {str(e)}", exc_info=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        logging.critical("Script failed unexpectedly during execution.", exc_info=True)
    finally:
        print("\nSubtitle Generation Script Finished.")