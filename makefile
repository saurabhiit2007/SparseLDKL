SRC=Train.cpp Predict.cpp Solver.cpp Evaluate.cpp Model.cpp Utils.cpp
HDR=Model.h Solver.h Evaluate.h Utils.h 
SOURCE=src/
INCLUDE=include/
LFLAGS=-Wall -O3 -msse -msse2 -msse3 -march=native -mtune=native -fPIC -ffast-math -ftree-vectorize
CFLAGS= -Wall -c -O3 -msse -msse2 -msse3 -march=native -mtune=native -fPIC -ffast-math -ftree-vectorize
DEBUG= -g
CC=g++
all: Train Predict

Model.o: $(SOURCE)Model.cpp $(INCLUDE)Model.h
	$(CC) $(CFLAGS) $(SOURCE)Model.cpp
Utils.o: $(SOURCE)Utils.cpp $(INCLUDE)Utils.h
	$(CC) $(CFLAGS) $(SOURCE)Utils.cpp
Solver.o: $(SOURCE)Solver.cpp $(INCLUDE)Solver.h Utils.o
	$(CC) $(CFLAGS) $(SOURCE)Solver.cpp
Evaluate.o: $(SOURCE)Evaluate.cpp $(INCLUDE)Evaluate.h Utils.o Model.o
	$(CC) $(CFLAGS) $(SOURCE)Evaluate.cpp	
Train.o: $(SOURCE)Train.cpp Solver.o Utils.o Model.o
	$(CC) $(CFLAGS) $(SOURCE)Train.cpp	
Predict.o: $(SOURCE)Predict.cpp Utils.o Model.o Evaluate.o
	$(CC) $(CFLAGS) $(SOURCE)Predict.cpp
Train: Train.o
	$(CC) $(LFLAGS) Train.o Model.o Solver.o Utils.o -o Train
Predict: Predict.o
	$(CC) $(LFLAGS) Predict.o Evaluate.o Model.o Utils.o -o Predict
clean:
	\rm -f *.o Train Predict

