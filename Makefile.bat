@echo off
REM Set Variables
set CXX=cl.exe
set SRC=src
set INC=include
if "%1" == "" 		goto x86
if /i %1 == x86		goto x86
if /i %1 == amd64	goto amd64
if /i %1 == clean	goto clean
REM Set PATH VARIABLE for Compiler
:x86
set ARC=x86
goto Compile
:amd64
set ARC=amd64
goto Compile
REM Compile the source
:Compile
call "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" %ARC%
%CXX% /c /O2 %SRC%\Model.cpp
%CXX% /c /O2 %SRC%\Utils.cpp
%CXX% /c /O2 %SRC%\Solver.cpp
%CXX% /c /O2 %SRC%\Evaluate.cpp
%CXX% /O2 %SRC%\Train.cpp Model.obj Solver.obj Utils.obj
%CXX% /O2 %SRC%\Predict.cpp Model.obj Utils.obj Evaluate.obj
goto :eof
:clean
del *.obj
del *.exe
goto :eof