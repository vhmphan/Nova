#!/bin/bash
g++ -o Bin/pack_Xrad.o -c pack_Xrad.cpp

g++ -o Bin/0_nova.o -c 0_nova.cpp
g++ -o Bin/0_nova Bin/0_nova.o 

g++ -o Bin/1_gamma.o -c 1_gamma.cpp
g++ -o Bin/1_gamma Bin/pack_Xrad.o Bin/1_gamma.o 
