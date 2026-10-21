@echo off
setlocal
cd /d "%~dp0"
if not exist out mkdir out
javac --add-modules jdk.httpserver -d out src\main\java\vectordb\VectorDbServer.java
if errorlevel 1 exit /b 1
java --add-modules jdk.httpserver -cp out vectordb.VectorDbServer
