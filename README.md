Python bindings for whisper.cpp

My fork version of whisper.cpp.py.
- Providing time stamps
- Providing word-level info

===============================
info from original authors are below.
===============================

<a href="https://www.buymeacoffee.com/lukeFxC" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

---
`pip install git+https://github.com/stlukey/whispercpp.py`

```python
from whispercpp import Whisper

w = Whisper('tiny')

result = w.transcribe("myfile.mp3")
text = w.extract_text(result)
```

Note: default parameters might need to be tweaked.
See Whispercpp.pyx.
