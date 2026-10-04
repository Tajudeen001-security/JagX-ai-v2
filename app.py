"""JagX AI v7.0.0 loader. Created by JagX & JRILICENSE."""
import gzip, base64, pathlib
_p = pathlib.Path(__file__).with_name("app_payload.b64")
exec(gzip.decompress(base64.b64decode(_p.read_text())).decode("utf-8"), globals())
