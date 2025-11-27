from ast import Tuple


base_config: Tuple = {
    "Co-Host-Speaker-Voice": "",
    "Host-Speaker-Voice": "",

    "Small-Text-Model": {
        "provider": {
            "name": "",
            "key": ""
        },
        "model": ""
    },

    "Big-Text-Model": {
        "provider": {
            "name": "",
            "key": ""
        },
        "model": ""
    },

    "Text-To-Speech-Model": {
        "provider": {
            "name": "",
            "endpoint": "",
            "key": ""
        },
        "model": "",
        "audio_format": ""
    },

    "Step1": {
        "max_tokens": 1028,
        "temperature": 0.7,
        "chunk_size": 1000,
        "max_chars": 100000
    },

    "Step2": {
        "max_tokens": 8126,
        "temperature": 1
    },

    "Step3": {
        "max_tokens": 8126,
        "temperature": 1
    }
}