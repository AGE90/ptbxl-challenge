#!/usr/bin/env bash
# Download the PTB-XL dataset (records100 + metadata) into data/raw/.
#
# Fetches PhysioNet's single bulk ZIP for the whole project instead of
# recursively wget-ing ~44,000 individual record files one at a time (the
# notebook's original approach) - that per-file HTTP overhead is what makes
# the naive download so slow. The ZIP also contains records500 (500Hz, full
# resolution), which this project doesn't use, so only records100 and the
# two metadata CSVs are extracted from it.
set -euo pipefail

VERSION="1.0.3"
ZIP_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-${VERSION}.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
ZIP_PATH="${RAW_DIR}/ptb-xl-${VERSION}.zip"
EXTRACT_DIR="${RAW_DIR}/.ptb-xl-extract"
TARGET_RECORDS_DIR="${RAW_DIR}/physionet.org/files/ptb-xl/${VERSION}/records100"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

for tool in curl unzip; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: ${tool} is required but not installed" >&2
        exit 1
    fi
done

if [[ "${FORCE}" -eq 0 && -f "${RAW_DIR}/ptbxl_database.csv" && -d "${TARGET_RECORDS_DIR}" ]]; then
    echo "PTB-XL data already present under ${RAW_DIR} - nothing to do (pass --force to re-download)."
    exit 0
fi

mkdir -p "${RAW_DIR}"

echo "Downloading PTB-XL ${VERSION} (~1.8 GB, single archive - resumable if interrupted)..."
curl --fail --location --continue-at - --output "${ZIP_PATH}" "${ZIP_URL}"

echo "Extracting metadata and records100 (skipping records500)..."
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
unzip -q "${ZIP_PATH}" "*ptbxl_database.csv" "*scp_statements.csv" "*records100/*" -d "${EXTRACT_DIR}"

SRC_ROOT="$(dirname "$(find "${EXTRACT_DIR}" -name ptbxl_database.csv -print -quit)")"

mkdir -p "$(dirname "${TARGET_RECORDS_DIR}")"
rm -rf "${TARGET_RECORDS_DIR}"
mv "${SRC_ROOT}/ptbxl_database.csv" "${RAW_DIR}/"
mv "${SRC_ROOT}/scp_statements.csv" "${RAW_DIR}/"
mv "${SRC_ROOT}/records100" "${TARGET_RECORDS_DIR}"

rm -rf "${EXTRACT_DIR}" "${ZIP_PATH}"

echo "Done. Data available under ${RAW_DIR}"
