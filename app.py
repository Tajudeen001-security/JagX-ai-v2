"""JagX AI 7.0.0 full source loader. Created by JagX & JRILICENSE."""
from pathlib import Path
_base = Path(__file__).parent
_parts = sorted(_base.glob("src_part*.txt"), key=lambda p: int(p.stem.replace("src_part", "")))
_src = "".join(p.read_text() for p in _parts)
exec(_src, globals())
