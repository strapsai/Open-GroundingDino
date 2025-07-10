python3 tools/inference_on_a_dir.py \
    -c tools/GroundingDINO_SwinT_OGC.py \
    -p /ocean/projects/cis240120p/emilian/triage_data/models/open-gdino-outputs/f8-v1-10k_80_1/checkpoint_best_regular.pth \
    -d /ocean/projects/cis240120p/emilian/triage_data/darpa/images \
    -t "blood near a person" \
    -o /ocean/projects/cis240120p/emilian/triage_data/darpa/preds/swint_ft_f8v1_db10k_80_1