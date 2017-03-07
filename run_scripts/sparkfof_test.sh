#!/bin/sh
#BSUB -J spark-fof-test
#BSUB -W 1:00 
#BSUB -o logs/spark-fof-test-%J.log
#BSUB -n 1
#BSUB -R rusage[mem=16000]

cd ~/Projects/spark-fof/tests
SPARK_HOME=$HOME/spark pytest
