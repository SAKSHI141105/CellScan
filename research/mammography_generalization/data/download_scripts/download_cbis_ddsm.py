"""CBIS-DDSM setup notes — this is a placeholder, not a working downloader.

The dataset is hosted on TCIA and requires the NBIA Data Retriever (or the
`tcia_utils` package) plus a manifest file, not a plain HTTP GET — there's
no anonymous bulk-download endpoint, so there's nothing meaningful to
automate here without a TCIA account in the loop.

Manual steps:
  1. https://www.cancerimagingarchive.net/collection/cbis-ddsm/ -> download
     the .tcia manifest (calcification + mass, train + test).
  2. Open the manifest in the NBIA Data Retriever app, point it at
     data/raw/cbis_ddsm/ as the output directory.
  3. Also grab the accompanying CSVs (calc_case_description_*.csv,
     mass_case_description_*.csv) from the same collection page — these
     carry the pathology labels and ROI mask file paths that
     src/data/dataset.py expects to join against.
  4. Run `python -m src.data.build_cbis_manifest` (not implemented yet —
     TODO) to flatten the DICOM tree + CSVs into the
     data/processed/cbis_ddsm/{train,val}.csv format configs/*.yaml expect:
     columns [image_path, mask_path, label, patient_id].
"""

if __name__ == "__main__":
    raise SystemExit(
        "CBIS-DDSM isn't a scriptable download (TCIA requires the NBIA Data "
        "Retriever + a manifest file). See the module docstring for the manual steps."
    )
