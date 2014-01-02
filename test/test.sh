#!/bin/sh
./test_my_classifier     /mnt/test-dataset/mit-cmu-png.txt /mnt/test-dataset/mit-cmu/abc-ground-truth.txt /mnt/classifiers/strongHypothesis-my.txt     /home/ramiro/roc-my.txt
./test_pavani_classifier /mnt/test-dataset/mit-cmu-png.txt /mnt/test-dataset/mit-cmu/abc-ground-truth.txt /mnt/classifiers/strongHypothesis-pavani.txt /home/ramiro/roc-pavani.txt
./test_vj_classifier     /mnt/test-dataset/mit-cmu-png.txt /mnt/test-dataset/mit-cmu/abc-ground-truth.txt /mnt/classifiers/strongHypothesis-vj.txt     /home/ramiro/roc-vj.txt
