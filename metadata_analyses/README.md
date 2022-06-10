# Analyse the metadata of Red Kite images from different sources

[getFileMeta.sh](getFileMeta.sh) will extract information from images of the three analysed sources using the embedded exif information. General information is extracted from a raw, unchanged version. The images were then manually tagged with relevant analyses aspects (see below). This information is also extracted using the same script.

The resulting [imageMetaData.csv](data/metadata_file/imageMetaData.csv) and [imageTags.csv](data/metadata_file/imageTags.csv) are evaluated using [imageMetadataAnalyses.Rmd](imageMetadataAnalyses.Rmd) R markdown script. The results can be seen in [imageMetadataAnalyses.md](figures/imageMetadataAnalyses.md).

## Image tags

 - duplicates (d)
    - duplicates are images that are taken by the same user of the same bird in the same observation session. While they increase the number of observations and the detail of observations they may lead to an overestimation of total birds.
 - quality (q)
    - any low quality images like blurred images or images with very small birds (e.g. far away so the the species cannot be easily distinguished anymore) are marked
 - no. of bird (n)
    - images with more than one Red Kite are marked
 - sitting (s)
    - if images feature sitting birds (i.e. ground touching (also for hunting) or any other posture that is not flying or floating, whatever that may be)

### Deprecated tags

 - archetype (a)
   - flying bird with sky or cloud background
 - background (g, u)
    - the background of bird images may be crucial for machine learning. The most common setting are images with a blue or cloudy sky background. Any deviation is marked by g for ground facing images and u for upward facing images with a noisy background like tree branches.
 - size
    - images with very small birds (e.g. far away) are marked
 - special images (spec)
    - if the images feature special images such as images of feathers or images of display scenes featuring an image of a Red Kite
 - processing (p)
    - any notable post-processing like watermarks or collages is recorded
    
## Annotation Procedure

Duplicates are identified automatically if a user uploads multiple images within one hour. A duplicate tag with the session-id is automatically added to the respective images (see [script](find_clusters.py)).

These automatic duplicates are then manually verified and any additional tags are added as follows: Import the images to Shotwell: https://wiki.ubuntuusers.de/Shotwell/, activate `save tags to file` in the settings. Use `Strg + M` to adapt or `Strg + T` to add keywords to the image.

Image tags are stored in the [IPTC](https://en.wikipedia.org/wiki/IPTC_Information_Interchange_Model) `keywords` field.
