from typing import Dict, Any, List, Optional, Literal
from elevenlabs.client import ElevenLabs
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
from elevenlabs import save
from google import genai
import time


FormatType = Literal[
    "podcast", "interview", "panel-discussion", "debate",
    "summary", "narration", "storytelling", "explainer",
    "lecture", "tutorial", "q-and-a","news-report",
    "executive-brief", "meeting", "analysis",
    "three-people-podcast", "three-people-panel-discussion",
    "three-people-debate", "four-people-podcast",
    "four-people-panel-discussion", "four-people-debate",
    "five-people-podcast", "five-people-panel-discussion", "five-people-debate"
]

SingleSpeakerFormats = Literal[
    "summary", "narration", "storytelling", "explainer",
    "lecture", "tutorial", "news-report", "executive-brief", "analysis"
]
SINGLE_SPEAKER_FORMATS = {
    "summary", "narration", "storytelling", "explainer",
    "lecture", "tutorial", "news-report", "executive-brief", "analysis"
}

TwoSpeakerFormats = Literal[
    "podcast", "interview", "panel-discussion",
    "debate", "q-and-a", "meeting"
]
TWO_SPEAKER_FORMATS = {
    "podcast", "interview", "panel-discussion",
    "debate", "q-and-a", "meeting"
}

ThreeSpeakerFormats = Literal[
    "three-people-podcast", "three-people-panel-discussion", "three-people-debate"
]
THREE_SPEAKER_FORMATS = {
    "three-people-podcast", "three-people-panel-discussion", "three-people-debate"
}

FourSpeakerFormats = Literal[
    "four-people-podcast", "four-people-panel-discussion", "four-people-debate"
]
FOUR_SPEAKER_FORMATS = {
    "four-people-podcast", "four-people-panel-discussion", "four-people-debate"
}

FiveSpeakerFormats = Literal[
    "five-people-podcast", "five-people-panel-discussion", "five-people-debate"
]
FIVE_SPEAKER_FORMATS = {
    "five-people-podcast", "five-people-panel-discussion", "five-people-debate"
}


LengthType = Literal["short", "medium", "long", "very-long"]


StyleType = Literal["normal", "friendly", "professional", "academic", "casual", "technical", "gen-z", "funny"]

SkipToOptions = [None, 1, 2, 3, 4]

def wait_for_next_step(seconds: float = 2):
    time.sleep(seconds)


def set_provider(
    provider_name: Optional[Literal['openai', 'lmstudio', 'ollama', 'groq', 'azure', 'google', 'anthropic', 'elevenlabs', 'custom']] = None,
    config: Optional[Dict[str, Any]] = None
):
    if provider_name is None:
        if config and "name" in config:
            provider_name = config["name"]
        else:
            raise ValueError("Provider name must be specified either directly or in config.")
    
    api_key = config.get("key") if config else None
    
    if provider_name == "openai":
        if api_key is None:
            raise ValueError("API key is required for OpenAI provider.")
        client = OpenAI(
            api_key=api_key,
        )
        return client
    elif provider_name == "lmstudio":
        client = OpenAI(
            base_url='http://localhost:1234/v1',
            api_key=api_key,
        )
        return client
    elif provider_name == "ollama":
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key=api_key,
        )
        return client
    elif provider_name == "groq":
        client = OpenAI(
            base_url='https://api.groq.com/openai/v1',
            api_key=api_key,
        )
        return client
    elif provider_name == "azure":
        if not config:
            raise ValueError("Config is required for Azure provider.")
        base_url = config.get("endpoint")
        version = config.get("version")
        if base_url is None:
            raise ValueError("Base URL is required for AzureOpenAI provider.")
        if version is None:
            raise ValueError("Version is required for AzureOpenAI provider.")
        if api_key is None:
            raise ValueError("Key is required for AzureOpenAI provider.")
        client = AzureOpenAI(
            azure_endpoint=base_url,
            api_version=version,
            api_key=api_key
        )
        return client
    elif provider_name == "google":
        if api_key is None:
            raise ValueError("API key is required for Google provider.")
        client = genai.Client(api_key=api_key)
        return client
    elif provider_name == "anthropic":
        client = Anthropic(api_key=api_key)
        return client
    elif provider_name == "elevenlabs":
        if api_key is None:
            raise ValueError("API key is required for ElevenLabs provider.")
        client = ElevenLabs(api_key=api_key)
        return client
    elif provider_name == "custom":
        if not config:
            raise ValueError("Config is required for custom provider.")
        base_url = config.get("endpoint")
        if base_url is None:
            raise ValueError("Base URL is required for custom provider.")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        return client
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def generate_text(
    client: Any = None,
    messages: Optional[List[Dict]] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    if client is None:
        raise ValueError("Client is required")
    
    if messages is None or not messages:
        raise ValueError("Messages are required")
    
    if isinstance(client, genai.Client):
        system_message = None
        user_content = []
        
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            elif message.get("role") == "user":
                user_content.append({"role": "user", "parts": [message.get("content", "")]})
            elif message.get("role") == "assistant":
                user_content.append({"role": "model", "parts": [message.get("content", "")]})
        
        # Generate content with Google's client
        response = client.generate_content(
            model=model,
            contents=user_content,
            generation_config=genai.GenerationConfig(
                system_instruction=system_message,
                max_output_tokens=max_tokens,
                temperature=temperature
            ),
        )
        return response.text
    elif isinstance(client, Anthropic):
        system_message = ""
        anthropic_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            elif message.get("role") in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": message.get("role"),
                    "content": message.get("content", "")
                })
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=anthropic_messages
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


def generate_speech(
    client: Any = None,
    text: str = None,
    voice: str = "alloy",
    model_name: str = "tts-1",
    response_format: str = "wav",
    output_path: str = "output"
):
    if isinstance(client, ElevenLabs):
        file_extension = response_format.split('_')[0].split('-')[0]
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice,
            model_id=model_name,
            output_format=response_format,
        )
        save(audio=audio, filename=str(f"{output_path}.{file_extension}"))
    else:
        with client.audio.speech.with_streaming_response.create(
            model=model_name,
            voice=voice,
            input=text,
            response_format=response_format
        ) as response:
            response.stream_to_file(str(f"{output_path}.{response_format}"))
    
    return f"{output_path}.{response_format}"