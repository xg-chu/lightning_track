wget https://github.com/Purkialo/GPAvatar/releases/download/resources/resources.tar ./resources.tar
tar -xvf resources.tar
mv resources/emoca/* ./engines/emoca/assets/
mv resources/FLAME/* ./engines/FLAME/assets/
mv resources/human_matting/* ./engines/human_matting/assets/
mv resources/mica/* ./engines/mica/assets/
rm -r resources/
