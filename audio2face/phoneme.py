# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import math
from collections import OrderedDict
import numpy as np

from audio2face import utils

CHARSIU_PHONEME_FPS = 100  # 100 phonemes per second

CHARSIU_LANG_MAP = {
    "en": "charsiu/en_w2v2_fc_10ms",
    "zh": "charsiu/zh_xlsr_fc_10ms",
}


class Phonemes:
    def __init__(self, data: np.ndarray, samplerate: float, lang: str) -> None:
        self.data = data
        self.samplerate = samplerate  # phonemes per second
        self.lang = lang

    def sec_to_sample(self, sec: float) -> int:
        true_samplerate = len(self.data) / (len(self.data) / self.samplerate + 0.02)  # due to edge effects
        return int(math.floor(sec * true_samplerate))

    def get_padded_buffer(self, ofs: int, length: int, pad_token_idx: int) -> np.ndarray:
        if ofs >= 0 and ofs + length <= len(self.data):
            return self.data[ofs : ofs + length]
        res = np.zeros((length, *self.data.shape[1:]), dtype=self.data.dtype)
        res[..., pad_token_idx] = 1.0
        begin = max(0, -ofs)
        end = min(length, len(self.data) - ofs)
        if begin < end:
            res[begin:end] = self.data[ofs + begin : ofs + end]
        return res

    # TODO Currently works only if new_samplerate is a factor of self.samplerate
    def resample(self, new_samplerate: float) -> None:
        if self.samplerate == new_samplerate:
            return
        skip = int(round(self.samplerate / new_samplerate))
        self.data = self.data[::skip]
        self.samplerate = new_samplerate


class PhonemeDetector:
    def __init__(self, langs: list[str], temperature: float = 1.0, torch_cache_root: str | None = None) -> None:
        self.langs = langs

        if len(self.langs) == 0:
            self.models = OrderedDict()
            self.num_phonemes: OrderedDict[str, int] = OrderedDict()
            self.max_num_phonemes = 0
            return

        logging.info(f"Using languages {self.langs} to initialize Phoneme Detector...")
        self.prepare_for_charsiu_import(torch_cache_root)
        from audio2face.deps.charsiu.src.Charsiu import charsiu_predictive_aligner

        self.models: OrderedDict[str, charsiu_predictive_aligner] = OrderedDict()
        self.num_phonemes: OrderedDict[str, int] = OrderedDict()
        for lang in self.langs:
            if lang not in CHARSIU_LANG_MAP.keys():
                raise ValueError(f'Unsupported phoneme lang: "{lang}". Should be in: {list(CHARSIU_LANG_MAP.keys())}')
            self.models[lang] = charsiu_predictive_aligner(CHARSIU_LANG_MAP[lang], temperature=temperature, lang=lang)
            self.num_phonemes[lang] = self.models[lang].charsiu_processor.processor.tokenizer.vocab_size
        self.max_num_phonemes = max(self.num_phonemes.values())

    def prepare_for_charsiu_import(self, torch_cache_root: str | None = None) -> None:
        utils.suppress_transformers_warnings()
        if torch_cache_root is not None:
            utils.change_huggingface_hub_cache_root(torch_cache_root)

    def lang_is_ready(self, lang: str) -> bool:
        return lang in self.models.keys()

    def validate_lang(self, lang: str) -> None:
        if not self.lang_is_ready(lang):
            raise ValueError(f'Unsupported phoneme lang: "{lang}". Should be in: {self.langs}')

    def get_sil_token_idx(self, lang: str) -> int:
        self.validate_lang(lang)
        return self.models[lang].charsiu_processor.processor.tokenizer.convert_tokens_to_ids("[SIL]")

    def gen_phonemes(self, audio_fpath: str, lang: str, new_samplerate: float | None = None) -> Phonemes:
        self.validate_lang(lang)
        phoneme_data = self.models[lang].align_probs(audio=audio_fpath)
        phonemes = Phonemes(phoneme_data, CHARSIU_PHONEME_FPS, lang)
        if new_samplerate is not None:
            phonemes.resample(new_samplerate)
        return phonemes
