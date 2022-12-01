
from pydub import AudioSegment            # 載入 pydub 的 AudioSegment 模組
from pydub.playback import play           # 載入 pydub.playback 的 play 模組

from playsound import playsound

# julia = AudioSegment.from_mp3("./Respond/re/cat1a.mp3")
# play(julia)

playsound("./respond/re/cat1a.mp3")
# playsound("./cat1a.mp3")