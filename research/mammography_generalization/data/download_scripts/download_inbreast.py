"""INbreast setup notes — placeholder, not a working downloader.

INbreast isn't publicly redistributable via direct link either; access goes
through a request form to the original authors (University of Porto /
INESC). Once you have the archive:

  1. Extract to data/raw/inbreast/ — expect AllDICOMs/, AllXML/ (lesion
     contours as XML, not raster masks), and INbreast.csv (BI-RADS + density).
  2. XML contours need rasterizing into binary masks aligned to each DICOM's
     pixel grid before they're usable by src/data/dataset.py — TODO:
     write build_inbreast_masks.py using the XML ROI point lists direct
     from the plist-style XML (cv2.fillPoly per ROI, unioned per image).
  3. This dataset is intentionally used for zero-shot cross-dataset eval
     only in this project (see src/evaluation/evaluate_cross_dataset.py),
     not for training — it's small (410 images) and the point is testing
     generalization to it, not fitting it.
"""

if __name__ == "__main__":
    raise SystemExit(
        "INbreast requires requesting access from the original authors — "
        "see the module docstring for the manual steps once you have the archive."
    )
