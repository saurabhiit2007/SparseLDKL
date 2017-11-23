@echo off
mkdir Results
mkdir Models
set base=dataset/
set NRun=1

#Banana
set Name=banana
set D=3
set lW=0.01
set lP=0.1
set lT=0.01
set S=0.001
set Z=50
Train.exe -D %D% -lW %lW% -lT %lT% -lP %lP% -S %S% -N %NRun% -Z %Z% %base%%Name%.train Models/%Name%
Predict.exe Models/%Name% %base%%Name%.test -N %NRun% -R Results/%Name%_Result.txt

