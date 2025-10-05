set -e


DATA_DIR=${2}

wd=$(pwd)

mkdir  $wd/output

python3 train_flow_b2.py \
  --data-dir   "$DATA_DIR"   \
  --output-dir $wd/output/  \
  --batch-size   16           \
  --ae-epochs    80




