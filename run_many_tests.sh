#!/bin/bash

config="_50k_1e-5_KGC_"


for i in {0..13}
do
  for j in {0..13}
  do
    if [ "$j" -gt "$i" ];
    then
      rm -r logs/
      python codetect_obj.py --logProgress --holdout1=$i --holdout2=$j
      python codetect_obj.py --mode=evaluate --holdout1=$i --holdout2=$j
      cp progressWhileTraining.csv  "progressWhileTraining"$config$i"_"$j".csv"
      cp classifications.csv  "classifications"$config$i"_"$j".csv"
    fi
  done
done
