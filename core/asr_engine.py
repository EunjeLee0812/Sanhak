from typing import Optional, List, Dict, Any
from faster_whisper import WhisperModel

class ASR:
    def __init__(self, model_size: str, device: str, compute_type: str, initial_prompt: str):
        # 메인에서 넘겨받은 설정값들을 저장하고 모델을 로드합니다.
        self.korean_only_prompt = initial_prompt
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, path: str, lang: str, beam_size: int, hotwords: Optional[List[str]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "language": lang,
            "beam_size": beam_size,
            "initial_prompt": self.korean_only_prompt,
            "vad_filter": True
        }
        
        if hotwords and len(hotwords) > 0:
            kwargs["hotwords"] = ",".join(hotwords)

        segs, _ = self.model.transcribe(path, **kwargs)
        return "".join(s.text for s in segs).strip()