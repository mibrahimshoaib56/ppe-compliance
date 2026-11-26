import os
import tempfile

d_tmp = "D:\\tmp"
if not os.path.exists(d_tmp):
    try:
        os.makedirs(d_tmp, exist_ok=True)
    except Exception:
        pass

if os.path.exists(d_tmp):
    os.environ["TEMP"] = d_tmp
    os.environ["TMP"] = d_tmp
    tempfile.tempdir = d_tmp
