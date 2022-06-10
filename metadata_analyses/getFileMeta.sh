exiftool -csv  -r data/images/untagged/ > data/metadata_file/imageMetaData.csv

exiftool -csv  -r -Keywords -Directory data/images/tagged/ > data/metadata_file/imageTags.csv
