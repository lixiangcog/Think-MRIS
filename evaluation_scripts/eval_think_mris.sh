REASONING_MODEL_PATH="pretrained_models/Think-MRIS-7B"
SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"
TEST_DATA_PATH="anonymous/ReasonSeg_test"
OUTPUT_PATH="./evaluation_outputs/think_mris"
NUM_PARTS=1
BATCH_SIZE=50

mkdir -p ${OUTPUT_PATH}

for IDX in $(seq 0 $((NUM_PARTS - 1))); do
    python evaluation_scripts/evaluation_visionreasoner.py \
        --reasoning_model_path ${REASONING_MODEL_PATH} \
        --segmentation_model_path ${SEGMENTATION_MODEL_PATH} \
        --test_data_path ${TEST_DATA_PATH} \
        --output_path ${OUTPUT_PATH} \
        --idx ${IDX} \
        --num_parts ${NUM_PARTS} \
        --batch_size ${BATCH_SIZE}
done


