from .helpers import generate_text, wait_for_next_step, FormatType, LengthType, StyleType
from .prompts import map_step2_system_prompt
from typing import Any, Dict, Optional
import logging, pickle, time
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)

class TranscriptError(Exception):
    pass
class FileReadError(TranscriptError):
    pass
class TranscriptGenerationError(TranscriptError):
    pass
class InvalidParameterError(TranscriptError):
    pass

def read_input_file(filename: str) -> str:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    content = file.read()
                logger.info(f"Successfully read file using {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
        raise FileReadError(f"Could not decode file '{filename}' with any common encoding")
    except FileNotFoundError:
        raise FileReadError(f"File '{filename}' not found")
    except IOError as e:
        raise FileReadError(f"Could not read file '{filename}': {str(e)}")

def generate_transcript(
    client,
    model_name,
    input_text,
    length,
    style,
    format_type,
    preference_text,
    system_prompt,
    max_tokens,
    temperature,
    chunk_token_limit,
    overlap_percent
) -> str:
    try:
        wait_for_next_step()
        
        # More conservative token estimation (3.5 chars per token)
        estimated_tokens = len(input_text) // 3.5
        
        # If input is likely too long, split it into chunks
        if estimated_tokens > chunk_token_limit:
            # Convert token count to character count for chunking
            chunk_size = int(chunk_token_limit * 3.5)
            overlap_size = int(chunk_size * overlap_percent / 100)
            
            # Create chunks with overlap
            chunks = []
            start = 0
            while start < len(input_text):
                end = min(start + chunk_size, len(input_text))
                chunks.append(input_text[start:end])
                start = end - overlap_size if end < len(input_text) else end
            
            logger.info(f"Input split into {len(chunks)} chunks with {overlap_percent}% overlap (chunk_token_limit: {chunk_token_limit})")
            
            # First chunk - generate the beginning of the transcript
            short_system_prompt = f"Create a {length} {style} {format_type} transcript. {preference_text}"
            
            conversation = [
                {"role": "system", "content": short_system_prompt},
                {"role": "user", "content": f"Create the beginning of a {format_type} transcript based on this content (part 1/{len(chunks)}): {chunks[0]}"},
            ]
            
            transcript = generate_text(
                client=client,
                model=model_name,
                messages=conversation,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Process remaining chunks with tqdm progress bar
            for i, chunk in tqdm(enumerate(chunks[1:], 2), total=len(chunks)-1, desc="Processing chunks"):
                # Add delay between API calls
                time.sleep(3)
                
                # Very minimal prompt to save tokens
                conversation = [
                    {"role": "system", "content": f"Continue the {format_type} transcript without repeating introductions."},
                    {"role": "user", "content": f"Continue the transcript with part {i}/{len(chunks)}: {chunk}"}
                ]
                
                next_part = generate_text(
                    client=client,
                    model=model_name,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                transcript += "\n" + next_part
            
            return transcript
        else:
            if system_prompt == None:
                # Original behavior for texts that fit within token limits
                system_prompt = map_step2_system_prompt(length=length, style=style, format_type=format_type, preference_text=preference_text)
            else:
                system_prompt = system_prompt

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]
            return generate_text(
                client=client,
                model=model_name,
                messages=conversation,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    except Exception as e:
        raise TranscriptGenerationError(f"Failed to generate transcript: {str(e)}")

def step2(
    client: Any = None,
    config: Optional[Dict[str, Any]] = None,
    input_file: str = None,
    output_dir: str = None,
    format_type: FormatType = "podcast",
    length: LengthType = "medium",
    style: StyleType = "normal",
    preference_text: str = "nothing",
    system_prompt: str = None
) -> str:
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Reading input file: {input_file}")
        input_text = read_input_file(input_file)
        
        logger.info(f"Generating {length} {style} transcript...")
        transcript = generate_transcript(
            client=client,
            model_name=config["Big-Text-Model"]["model"],
            input_text=input_text,
            format_type=format_type,
            length=length,
            style=style,
            system_prompt=system_prompt,
            preference_text=preference_text,
            max_tokens=config["Step2"]["max_tokens"],
            temperature=config["Step2"]["temperature"],
            chunk_token_limit=config["Step2"].get("chunk_token_limit", 2000),
            overlap_percent=config["Step2"].get("overlap_percent", 10)
        )

        output_file = output_dir / 'data'
        with open(f"{output_file}.pkl", 'wb') as file:
            pickle.dump(transcript, file)
        with open(f"{output_file}.txt", 'w') as file:
            file.write(transcript)

        logger.info(f"Transcript saved to: {output_file}")
        return str(input_file), str(f"{output_file}.pkl")

    except (FileReadError, TranscriptGenerationError, InvalidParameterError) as e:
        logger.error(f"Transcript generation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during transcript generation: {str(e)}")
        raise TranscriptError(f"Transcript generation failed: {str(e)}")