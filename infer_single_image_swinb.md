python tools/inference_on_a_image.py \
    -c tools/GroundingDINO_SwinT_OGC.py \
    -p /ocean/projects/cis240120p/emilian/triage_data/models/groundingdino_swint_ogc.pth \
    -i /ocean/projects/cis240120p/emilian/triage_data/darpa/images/1727638201466528892.jpg \
    -t "blood near a person" \
    -o inference_output/trials