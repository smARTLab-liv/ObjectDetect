# ObjectDetect

The code in this repository provides an implementation for the methods discussed in the following paper: [Fully Convolutional One-Shot Object Segmentation for Industrial Robotics](https://arxiv.org/pdf/1903.00683.pdf). The paper is dedicated to the memory of our wonderful colleague and friend Benjamin Schnieders, a bright young scientist, who recently passed away.

## Usage:

To run the program using the ObjectDetection@Work dataset, a preprocessed dataset can be obtained from:

[Mirror 1](https://cgi.csc.liv.ac.uk/~gpalmer/ObjectDetectionAtWork_Preprocessed.tar.gz)
Filesize: 11.9 GB
md5sum: b6f2e2ee7ad747eeb4edaa4a7c2593aa

Upon downloading an extracting the dataset, the framework can be run directly via:

	python codetect_obj.py --logProgress --holdout1=0 --holdout2=1

where holdout1 and holdout2 represent the indexes belonging to the holdout objects.

Alternatively, the following script can be run to test each holdout combination:

        run_many_tests.sh
