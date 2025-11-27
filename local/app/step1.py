from .helpers import generate_text, FormatType, wait_for_next_step
from typing import Optional, List, Dict, Any
from .prompts import step1_prompt
import logging, PyPDF2, os, time
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)

class PDFProcessingError(Exception):
    pass
class PDFValidationError(PDFProcessingError):
    pass
class PDFExtractionError(PDFProcessingError):
    pass
class ChunkProcessingError(PDFProcessingError):
    pass

def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        raise PDFValidationError(f"File not found at path: {file_path}")
    if not file_path.lower().endswith('.pdf'):
        raise PDFValidationError("File is not a PDF")
    return True

def get_pdf_metadata(file_path: str) -> Optional[dict]:
    try:
        if not validate_pdf(file_path):
            return None

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata
            }
            return metadata
    except PDFValidationError as e:
        raise e
    except Exception as e:
        raise PDFExtractionError(f"Failed to extract metadata: {str(e)}")

def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> str:
    try:
        if not validate_pdf(file_path):
            return None

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF with {num_pages} pages")

            extracted_text = []
            total_chars = 0

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text() or ""

                if total_chars + len(text) > max_chars:
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    logger.info(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break

                extracted_text.append(text)
                total_chars += len(text)
                logger.debug(f"Processed page {page_num + 1}/{num_pages}")

            final_text = '\n'.join(extracted_text)
            logger.info(f"Extraction complete. Total characters: {len(final_text)}")
            return final_text

    except PDFValidationError as e:
        raise e
    except PyPDF2.PdfReadError as e:
        raise PDFExtractionError(f"Invalid or corrupted PDF file: {str(e)}")
    except Exception as e:
        raise PDFExtractionError(f"Failed to extract text: {str(e)}")

def create_word_bounded_chunks(text: str, target_chunk_size: int) -> List[str]:
    try:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > target_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
    except Exception as e:
        raise ChunkProcessingError(f"Failed to create text chunks: {str(e)}")

def process_chunk(
        client,
        text_chunk,
        system_prompt,
        chunk_num,
        model_name,
        max_tokens,
        temperature,
        format_type
    ) -> str:
    try:
        wait_for_next_step()
        if system_prompt == None:
            system = step1_prompt.format(text_chunk=text_chunk, format_type=format_type)
        else:
            system = system_prompt
            
        messages = [
            {"role": "user", "content": system},
        ]
        return generate_text(
            client=client,
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    except Exception as e:
        raise ChunkProcessingError(f"Failed to process chunk {chunk_num}: {str(e)}")

def step1(
    pdf_path: str,
    client: Any = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = None,
    format_type: FormatType = "podcast",
    system_prompt: str = None
) -> str:
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            metadata = get_pdf_metadata(pdf_path)
            if metadata:
                logger.info(f"PDF Pages: {metadata['num_pages']}")
        except PDFExtractionError as e:
            logger.warning(f"Failed to extract metadata: {e}")

        extracted_text = extract_text_from_pdf(pdf_path, config["Step1"]["max_chars"])
        if not extracted_text:
            raise PDFExtractionError("No text extracted from PDF")

        input_file = output_dir / 'extracted_text.txt'
        input_file.write_text(extracted_text, encoding='utf-8')

        chunks = create_word_bounded_chunks(extracted_text, config["Step1"]["chunk_size"])
        output_file = output_dir / f"clean_{input_file.name}"

        logger.info(f"Processing {len(chunks)} chunks")

        with open(output_file, 'w', encoding='utf-8') as out_file:
            for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks", disable=None)):
                processed_chunk = process_chunk(
                    client=client,
                    text_chunk=chunk,
                    chunk_num=chunk_num,
                    format_type=format_type,
                    system_prompt=system_prompt,
                    model_name=config["Small-Text-Model"]["model"],
                    max_tokens=config["Step1"]["max_tokens"],
                    temperature=config["Step1"]["temperature"]
                )
                out_file.write(processed_chunk + "\n")
                out_file.flush()

        logger.info("Processing complete")
        return str(output_file)

    except (PDFProcessingError, ChunkProcessingError) as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {str(e)}")
        raise PDFProcessingError(f"PDF processing failed: {str(e)}")