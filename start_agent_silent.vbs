Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d C:\Users\Administrator\Desktop\ClipPulse-AI && .\venv\Scripts\python.exe app.py", 0, False