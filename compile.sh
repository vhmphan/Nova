#!/bin/bash
g++ -o Bin/pack_Xrad.o -c pack_Xrad.cpp

g++ -o Bin/0_nova.o -c 0_nova.cpp
g++ -o Bin/0_nova Bin/pack_Xrad.o Bin/0_nova.o 
