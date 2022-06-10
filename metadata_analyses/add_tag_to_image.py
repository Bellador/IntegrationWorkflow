import pyexiv2
import glob
import pandas as pd
import re
from tqdm import tqdm

for f in tqdm(glob.glob('data/metadata_platform/*_sessions.csv')):
    images = pd.read_csv(f)

    m = re.compile(r'.*(flickr|ebird|inaturalist).*', re.I)
    platform = m.match(f).group(1)
    for row in tqdm(images.itertuples()):
        try:
            metadata = pyexiv2.ImageMetadata('data/images/tagged/' + platform + '/' + row.filename)

            metadata.read()
            key = 'Iptc.Application2.Keywords'

            keywords = {row.session_id}
            if key in metadata.iptc_keys:
                keywords.update(metadata[key].value)

            metadata[key] = list(keywords)

            metadata.write()
        except Exception as e:
            print(e)
