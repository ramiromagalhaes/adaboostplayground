#!/bin/sh
#for FILE_NBR in 020 040 060 080 100 120 140 160 180 200; do ./test_my_classifier /mnt/test-dataset/mit-cmu-png.txt /mnt/test-dataset/mit-cmu/abc-ground-truth.txt /home/ramiro/workspace/classifiers/strongHypothesis-my-version3-$FILE_NBR.txt /home/ramiro/roc-my-version3-$FILE_NBR.txt; done;

#for FILE_NBR in 020 040 060 080 100 120 140 160 180 200; do ./test_pavani_classifier /mnt/test-dataset/mit-cmu-png.txt /mnt/test-dataset/mit-cmu/abc-ground-truth.txt /home/ramiro/workspace/classifiers/strongHypothesis-pavani-$FILE_NBR.txt /home/ramiro/roc-pavani-$FILE_NBR.txt; done;

for FILE_NBR in 020 040 060 080 100 120 140 160 180 200; do ./test_vj_classifier /mnt/test-dataset/mit-cmu-png.txt /mnt/test-dataset/mit-cmu/abc-ground-truth.txt /home/ramiro/workspace/classifiers/strongHypothesis-vj-$FILE_NBR.txt /home/ramiro/roc-vj-$FILE_NBR.txt; done;
