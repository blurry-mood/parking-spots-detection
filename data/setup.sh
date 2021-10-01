wget http://cnrpark.it/dataset/CNR-EXT_FULL_IMAGE_1000x750.tar
pv CNR-EXT_FULL_IMAGE_1000x750.tar | tar -x
rm CNR-EXT_FULL_IMAGE_1000x750.tar 
mkdir dataset
mv *.csv dataset
mv FULL* dataset

python make_csv.py