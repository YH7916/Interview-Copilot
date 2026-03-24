@echo off
setlocal
set ROOT=%~dp0..
call "%ROOT%\.venv\Scripts\python.exe" -m nanobot.cli.commands interview-agent
