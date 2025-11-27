from .helpers import wait_for_next_step, generate_speech
from typing import List, Tuple, Dict, Any, Optional
import logging, pickle, ast, re
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)

class AudioGenerationError(Exception):
    pass

def load_podcast_data(data_path: Path) -> List[Tuple[str, str]]:
    try:
        with open(data_path, 'rb') as file:
            podcast_text = pickle.load(file)
        return ast.literal_eval(podcast_text)
    except FileNotFoundError:
        raise FileNotFoundError(f"Podcast data file not found: {data_path}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid podcast data format: {str(e)}")

def concatenate_audio_files(segment_dir: Path, format: str = "wav") -> Tuple[np.ndarray, int]:
    audio_files = sorted(
        segment_dir.glob(f"*podcast_segment_*.{format}"),
        key=lambda x: int(re.search(r'segment_(\d+)\.wav', str(x)).group(1))
    )
    
    if not audio_files:
        raise FileNotFoundError(f"No audio segments found in {segment_dir}")
    
    # Read the first file to get the sample rate
    first_data, sample_rate = sf.read(audio_files[0])
    audio_data = [first_data]
    
    # Read the rest of the files
    for file in audio_files[1:]:
        try:
            data, _ = sf.read(file)
            audio_data.append(data)
        except Exception as e:
            logger.error(f"Failed to read audio file {file}: {str(e)}")
            continue
    
    return np.concatenate(audio_data), sample_rate

def generate_speaker_audio(
    client,
    model_name,
    text,
    output_path,
    voice,
    response_format
) -> None:
    try:
        wait_for_next_step()
        generate_speech(client=client, model_name=model_name, text=text, output_path=output_path, voice=voice, response_format=response_format)
    except Exception as e:
        raise AudioGenerationError(f"Failed to generate audio: {str(e)}")

def parse_audio_format(format_string):
    parts = format_string.split('_')
    
    # Default values
    audio_format = parts[0]
    sample_rate = None
    bit_depth = None
    
    # Extract sample rate and bit depth if provided
    if len(parts) > 1 and parts[1].isdigit():
        sample_rate = int(parts[1])
    
    if len(parts) > 2 and parts[2].isdigit():
        bit_depth = int(parts[2])
    
    return audio_format, sample_rate, bit_depth

def step4(
    client: Any = None,
    config: Optional[Dict[str, Any]] = None,
    input_dir: str = None,
    output_dir: str = None
) -> Path:
    model_name = config["Text-To-Speech-Model"]["model"]
    host = config["Host-Speaker-Voice"]
    co_host_1 = config["Co-Host-Speaker-1-Voice"]
    co_host_2 = config["Co-Host-Speaker-2-Voice"]
    co_host_3 = config["Co-Host-Speaker-3-Voice"]
    co_host_4 = config["Co-Host-Speaker-4-Voice"]
    response_format = config["Text-To-Speech-Model"].get("audio_format", "wav")
    
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        audio_format, sample_rate, bit_depth = parse_audio_format(response_format)

        if audio_format not in ["wav", "mp3", "ogg", "flac", "aac"]:
            raise ValueError(f"Unsupported audio format: {audio_format}")
        
        # Create output directories
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Load podcast data
        podcast_data = load_podcast_data(input_dir / "podcast_ready_data.pkl")
        
        # Generate audio segments
        for i, (speaker, text) in enumerate(tqdm(podcast_data, desc="Generating podcast segments"), 1):
            output_path = segments_dir / f"podcast_segment_{i}"

            if speaker == "Speaker 1":
                current_voice = host
            elif speaker == "Speaker 3":
                current_voice = co_host_2
            elif speaker == "Speaker 4":
                current_voice = co_host_3
            elif speaker == "Speaker 3":
                current_voice = co_host_4
            else:
                current_voice = co_host_1

            generate_speaker_audio(
                client=client,
                text=text,
                model_name=model_name,
                output_path=output_path,
                voice=current_voice,
                response_format=response_format
            )
        
        # Concatenate all segments
        logger.info("Concatenating audio segments...")
        final_audio, detected_sample_rate = concatenate_audio_files(segments_dir, audio_format)
        
        # Save final podcast with the detected sample rate
        final_path = f"{output_dir}/podcast.{audio_format}"
        sf.write(str(final_path), final_audio, detected_sample_rate)
        logger.info(f"Podcast generated successfully at {final_path}")
        
        return final_path
        
    except Exception as e:
        logger.error(f"Failed to generate podcast: {str(e)}")
        raise