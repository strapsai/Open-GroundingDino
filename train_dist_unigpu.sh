GPU_NUM="1"
CFG="config/cfg_odvg.py"
DATASETS="config/datasets_mixed_odvg.json"
OUTPUT_DIR="/ocean/projects/cis240120p/emilian/triage_data/models/open-gdino-outputs/f8-v2-10k_80_1"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /ocean/projects/cis240120p/emilian/triage_data/models/groundingdino_swint_ogc.pth \
        --options text_encoder_type=/ocean/projects/cis240120p/emilian/triage_data/models/bert