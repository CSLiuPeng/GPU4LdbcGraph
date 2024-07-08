cd ../..
cd data/LdbcDataset

zstd -d cit-Patents.tar.zst	
tar -xvf cit-Patents.tar

zstd -d datagen-7_5-fb.tar.zst
tar -xvf datagen-7_5-fb.tar

zstd -d datagen-7_6-fb.tar.zst
tar -xvf datagen-7_6-fb.tar

zstd -d datagen-7_7-zf.tar.zst
tar -xvf datagen-7_7-zf.tar

zstd -d datagen-7_8-zf.tar.zst
tar -xvf datagen-7_8-zf.tar

zstd -d datagen-7_9-fb.tar.zst
tar -xvf datagen-7_9-fb.tar

zstd -d datagen-8_0-fb.tar.zst
tar -xvf datagen-8_0-fb.tar

zstd -d datagen-8_1-fb.tar.zst
tar -xvf datagen-8_1-fb.tar

zstd -d datagen-8_2-zf.tar.zst
tar -xvf datagen-8_2-zf.tar

zstd -d datagen-8_3-zf.tar.zst
tar -xvf datagen-8_3-zf.tar

zstd -d datagen-8_4-fb.tar.zst
tar -xvf datagen-8_4-fb.tar

mkdir zipfile
mv *.zst zipfile/
mv *.tar zipfile/