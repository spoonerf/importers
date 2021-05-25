"""executes bulk dataset import + chart updates for the UN_SDGs dataset.
Usage:
    python -m un_sdg.main
"""

from un_sdg import DATASET_NAMESPACE, DATASET_DIR, OUTPATH

from un_sdg import (
    download, 
    clean
)

from standard_importer import import_dataset, upsert_suggested_chart_revisions

def main():
    download.main()
    clean.main()
    import_dataset.main(DATASET_DIR, DATASET_NAMESPACE)

    match_variables.main()
    prepare_chart_updates.main()
    upsert_suggested_chart_revisions.main(DATASET_DIR)

if __name__ == '__main__':
    main()