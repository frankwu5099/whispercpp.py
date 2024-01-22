#!python
# cython: language_level=3

import ffmpeg
import numpy as np
import requests
import os
from pathlib import Path

MODELS_DIR = str(Path('~/.ggml-models').expanduser())
print("Saving models to:", MODELS_DIR)


cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* DEFAULT_MODEL = 'tiny'
cdef char* LANGUAGE = b'en'
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-tiny.en.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin',
    'ggml-base.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-small.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-medium.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-large.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
}

def model_exists(model):
    return os.path.exists(Path(MODELS_DIR).joinpath(model))

def download_model(model):
    if model_exists(model):
        return

    print(f'Downloading {model}...')
    url = MODELS[model]
    r = requests.get(url, allow_redirects=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(Path(MODELS_DIR).joinpath(model), 'wb') as f:
        f.write(r.content)


cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] load_audio(bytes file, int sr = SAMPLE_RATE):
    try:
        out = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le",
                acodec="pcm_s16le",
                ac=1, ar=sr
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True
            )
        )[0]
    except:
        raise RuntimeError(f"File '{file}' not found")

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    return frames

cdef whisper_full_params default_params() nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.language = <const char *> LANGUAGE
    n_threads = N_THREADS
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model=DEFAULT_MODEL, pb=None, buf=None):
        
        model_fullname = f'ggml-{model}.bin'
        download_model(model_fullname)
        model_path = Path(MODELS_DIR).joinpath(model_fullname)
        cdef bytes model_b = str(model_path).encode('utf8')
        
        if buf is not None:
            self.ctx = whisper_init_from_buffer(buf, buf.size)
        else:
            self.ctx = whisper_init_from_file(model_b)
        
        self.params = default_params()
        whisper_print_system_info()


    def __dealloc__(self):
        whisper_free(self.ctx)

    def transcribe(self, filename=TEST_FILE):
        print("Loading data..")
        if (type(filename) == np.ndarray) :
            temp = filename
        
        elif (type(filename) == str) :
            temp = load_audio(<bytes>filename)
        else :
            temp = load_audio(<bytes>TEST_FILE)

        
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = temp

        print("Transcribing..")
        return whisper_full(self.ctx, self.params, &frames[0], len(frames))
    
    def extract_text(self, int res):
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [
            whisper_full_get_segment_text(self.ctx, i).decode() for i in range(n_segments)
        ]

    def extract_timestamp(self, int res):
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [
            whisper_full_get_segment_t0(self.ctx, i) for i in range(n_segments)
        ], [
            whisper_full_get_segment_t1(self.ctx, i) for i in range(n_segments)
        ]

    def extract_token_level(self, int res):
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [[
            whisper_full_get_token_p(self.ctx, i, j) for j in range(whisper_full_n_tokens(self.ctx, i))
        ] for i in range(n_segments)
        ], [[
            whisper_full_get_token_text(self.ctx, i, j).decode() for j in range(whisper_full_n_tokens(self.ctx, i))
        ] for i in range(n_segments)]
    
    def extract_segments(self, int res):
        if res != 0:
            raise RuntimeError
        return {
            'segments': [
                {
                    'start': t0/100.,
                    'end': t1/100.,
                    'text': text,
                    'words': [
                        {'score':score, 'word': tok} for score, tok in zip(scores, toks) if '[_' not in tok
                    ]
                }
                for text, t0, t1, scores, toks in zip(self.extract_text(res), *self.extract_timestamp(res), *self.extract_token_level(res))
            ]
        }
