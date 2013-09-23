#!/bin/sh
./train_vj_classifier     /mnt/faces/faces.pgm /mnt/fundo/background-v2.pgm /mnt/features/haarwavelets-original.txt /mnt/classifiers/strongHypothesis-vj.txt 200;
./train_pavani_classifier /mnt/faces/faces.pgm /mnt/fundo/background-v2.pgm /mnt/features/haarclassifiers-optimized.txt /mnt/classifiers/strongHypothesis-pavani.txt 200;
./train_my_classifier     /mnt/faces/faces.pgm /mnt/fundo/background-v2.pgm /mnt/features/haarclassifiers-optimized.txt /mnt/classifiers/strongHypothesis-my.txt 200;
