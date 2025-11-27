import os
import gradio as gr
import argparse
from local_notebooklm.processor import podcast_processor
from local_notebooklm.steps.helpers import LengthType, FormatType, StyleType, SkipToOptions


def process_podcast(pdf_file, config_file, format_type, length, style, language, additional_preference, output_dir, skip_to):
    if pdf_file is None and (skip_to is None or skip_to == 1):
        return "Please upload a PDF file first.", None, "", "", ""
    
    if not output_dir:
        output_dir = "./local_notebooklm/web_ui/output"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        return f"Failed to create output directory: {str(e)}", None, "", "", ""
    
    try:
        if hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            pdf_path = pdf_file
        
        if config_file is None:
            config_path = "./example_config.json"
        else:
            if hasattr(config_file, 'name'):
                config_path = config_file.name
            else:
                config_path = config_file
        
        print(f"Processing with output_dir: {output_dir}")
        
        success, result = podcast_processor(
            pdf_path=pdf_path,
            config_path=config_path,
            format_type=format_type,
            length=length,
            style=style,
            preference=additional_preference if additional_preference else None,
            output_dir=output_dir,
            language=language,
            skip_to=skip_to
        )
        
        if success:
            audio_path = os.path.join(output_dir, "podcast.wav")
            file_contents = {}
            generated_files = [
                "step1/extracted_text.txt",
                "step1/clean_extracted_text.txt",
                "step2/data.pkl",
                "step3/podcast_ready_data.pkl",
                "step3/podcast_ready_data.txt"
            ]
            
            for file in generated_files:
                full_path = os.path.join(output_dir, file)
                if os.path.exists(full_path) and file.endswith(".txt"):
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            file_contents[file] = file_content[:1000] + "..." if len(file_content) > 1000 else file_content
                    except Exception as e:
                        file_contents[file] = f"Error reading file: {str(e)}"
            
            return "Audio Generated Successfully!", audio_path, file_contents.get("step1/extracted_text.txt", ""), file_contents.get("step1/clean_extracted_text.txt", ""), file_contents.get("step3/podcast_ready_data.txt", "")
        else:
            return f"Failed to generate audio: {result}", None, "", "", ""
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"An error occurred: {str(e)}\n\nDetails:\n{error_details}", None, "", "", ""

def create_gradio_ui():
    format_options = list(FormatType.__args__) if hasattr(FormatType, '__args__') else ["podcast"]
    length_options = list(LengthType.__args__) if hasattr(LengthType, '__args__') else ["medium"]
    style_options = list(StyleType.__args__) if hasattr(StyleType, '__args__') else ["conversational"]
    
    with gr.Blocks(title="Local-NotebookLM") as app:
        gr.Markdown("# 🎙️ Local-NotebookLM: PDF to Audio Converter")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                gr.Markdown("*Upload Config JSON (Optional) - Default: ./example_config.json*")
                config_file = gr.File(label="Config JSON", file_types=[".json"])
                format_type = gr.Dropdown(choices=format_options, label="Select Format", value=format_options[0])
                length = gr.Dropdown(choices=length_options, label="Select Length", value=length_options[1] if len(length_options) > 1 else length_options[0])
                style = gr.Dropdown(choices=style_options, label="Select Style", value=style_options[0])
                language = gr.Dropdown(
                    choices=["english", "german", "french", "spanish", "italian", "portuguese"],
                    label="Select Language",
                    value="english"
                )
                additional_preference = gr.Textbox(
                    label="Additional Preferences (Optional)",
                    placeholder="Focus on key points, provide examples, etc."
                )
                output_dir = gr.Textbox(
                    label="Output Directory", 
                    value="./local_notebooklm/web_ui/output",
                    placeholder="Enter the path where output files will be saved"
                )
                skip_to = gr.Dropdown(
                    choices=SkipToOptions,
                    label="Skip to Step (Optional)",
                    value=None,
                    info="Select a step to start from if you want to skip earlier steps"
                )
                generate_button = gr.Button("Generate Podcast")
            
            with gr.Column(scale=2):
                result_message = gr.Textbox(label="Status")
                audio_output = gr.Audio(label="Generated Podcast", type="filepath")
                
                with gr.Accordion("View Extracted Text", open=False):
                    extracted_text = gr.Textbox(label="Extracted Text", lines=10)
                
                with gr.Accordion("View Clean Extracted Text", open=False):
                    clean_text = gr.Textbox(label="Clean Extracted Text", lines=10)
                
                with gr.Accordion("View Podcast Script", open=False):
                    audio_script = gr.Textbox(label="Podcast Script", lines=15)
        
        gr.Markdown("---")
        gr.Markdown("Local-NotebookLM by Gökdeniz Gülmez")
        gr.Markdown("[GitHub Repository](https://github.com/Goekdeniz-Guelmez/Local-NotebookLM)")
        
        generate_button.click(
            fn=process_podcast,
            inputs=[pdf_file, config_file, format_type, length, style, language, additional_preference, output_dir, skip_to],
            outputs=[result_message, audio_output, extracted_text, clean_text, audio_script]
        )

    return app

def run_gradio_ui(share=False, port=None):
    app = create_gradio_ui()
    app.launch(share=share, server_port=port)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Local-NotebookLM web UI")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--port", type=int, default=None, help="Port to run the interface on")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    run_gradio_ui(share=args.share, port=args.port)

if __name__ == "__main__" or __name__ == "local_notebooklm.web_ui":
    main()