import os
import time
try:
    # For Linux/Unix systems
    os.system('for i in {1..5}; do echo -e "\a"; sleep 0.5; done')
    # For Windows systems
    os.system('powershell -c "(New-Object Media.SoundPlayer).PlaySync([System.IO.Path]::Combine([System.Environment]::SystemDirectory, \'media\\Windows Notify.wav\'))"')
    # Alternative method for Windows
    os.system('echo \x07\x07\x07\x07\x07')
except:
    # Fallback if above methods fail
    import sys
    for _ in range(5):
        sys.stdout.write('\a')
        sys.stdout.flush()
        time.sleep(0.5)