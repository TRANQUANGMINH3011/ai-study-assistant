from pathlib import Path

from .app.helpers import set_provider
from .app.step1 import step1
from .app.step2 import step2
from .app.step3 import step3
from .app.step4 import step4

def podcast_processor(
    pdf_path,
    config_path=None,
    format_config=None,
    format_type="summary",
    length="medium",
    style="normal",
    preference="nothing",
    output_dir="./output",
    skip_to: int = None,
    language: str = "english"
):
    # Load config
    if config_path:
        config = format_config
    elif config_path:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        from local.config import base_config
        config = base_config
    
    # Create output directories
    output_base = Path(output_dir)
    output_dirs = {
        "step1": output_base / "step1",
        "step2": output_base / "step2",
        "step3": output_base / "step3",
        "step4": output_base / "step4"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set up clients
    small_text_client = set_provider(config=config["Small-Text-Model"]["provider"])
    big_text_client = set_provider(config=config["Big-Text-Model"]["provider"])
    tts_client = set_provider(config=config["Text-To-Speech-Model"]["provider"])
    
    try:
        # Initialize variables for file paths that might be skipped
        cleaned_text_file = None
        transcript_file = None
        
        # Extract system prompts for each step (with fallbacks to general system prompt)
        system_prompts = {}
        for step_name in ["step1", "step2", "step3"]:
            # Check if there's a specific system prompt for this step
            if step_name in config and "system" in config[step_name]:
                system_prompts[step_name] = config[step_name]["system"]
            # Fall back to general system prompt if available
            elif "system" in config:
                system_prompts[step_name] = config["system"]
            else:
                system_prompts[step_name] = None
        
        # Step 1: Process PDF
        if not skip_to or skip_to <= 1:
            print("Step 1: Processing PDF...")
            cleaned_text_file = step1(
                client=small_text_client,
                pdf_path=pdf_path,
                config=config,
                output_dir=str(output_dirs["step1"]),
                system_prompt=system_prompts["step1"]
            )
        else:
            # If skipping, find the most recent output file from step1
            print("Skipping Step 1, looking for existing output...")
            step1_files = list(output_dirs["step1"].glob("*.txt"))
            if step1_files:
                cleaned_text_file = str(sorted(step1_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
                print(f"Using existing file from Step 1: {cleaned_text_file}")
            else:
                error_msg = "No output files found from Step 1. Cannot skip this step."
                print(error_msg)
                return False, error_msg
        
        # Step 2: Generate transcript
        if not skip_to or skip_to <= 2:
            print("Step 2: Generating transcript...")
            _, transcript_file = step2(
                client=big_text_client,
                config=config,
                input_file=cleaned_text_file,
                output_dir=str(output_dirs["step2"]),
                format_type=format_type,
                length=length,
                style=style,
                preference_text=preference,
                system_prompt=system_prompts["step2"]
            )
        else:
            # If skipping, find the most recent output file from step2
            print("Skipping Step 2, looking for existing output...")
            step2_files = list(output_dirs["step2"].glob("*.pkl"))
            if step2_files:
                transcript_file = str(sorted(step2_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
                print(f"Using existing file from Step 2: {transcript_file}")
            else:
                error_msg = "No output files found from Step 2. Cannot skip this step."
                print(error_msg)
                return False, error_msg
        
        # Step 3: Optimize for TTS
        if not skip_to or skip_to <= 3:
            print("Step 3: Optimizing for text-to-speech...")
            step3(
                client=big_text_client,
                config=config,
                input_file=transcript_file,
                output_dir=str(output_dirs["step3"]),
                format_type=format_type,
                system_prompt=system_prompts["step3"],
                language=language
            )
        else:
            print("Skipping Step 3, assuming files exist in output directory...")
        
        # Step 4: Generate audio
        if not skip_to or skip_to <= 4:
            print("Step 4: Generating audio...")
            final_audio_path = step4(
                client=tts_client,
                config=config,
                input_dir=str(output_dirs["step3"]),
                output_dir=str(output_dirs["step4"])
            )
            
            print(f"Podcast generation complete! Final audio file: {final_audio_path}")
            return True, final_audio_path
        
        return True, "Process completed successfully (without audio generation)"
        
    except Exception as e:
        error_msg = f"Error during podcast generation: {str(e)}"
        print(error_msg)
        return False, error_msg