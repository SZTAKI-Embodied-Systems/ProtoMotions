https://nvlabs.github.io/ProtoMotions/getting_started/kimodo_preparation.html

# Convert CSV to ProtoMotion .motion format (KIMODO)
python data/scripts/convert_g1_csv_to_proto.py \
    --input-dir data/kimodo_output/ \
    --output-dir data/kimodo_output_proto \
    --input-fps 30 \
    --output-fps 30 \
    --pos-units m \
    --rot-format quat_wxyz \
    --joint-units rad \
    --no-has-header \
    --no-has-frame-column \
    --force-remake

# Convert CSV to ProtoMotion .motion format (BONES_SEED)
python data/scripts/convert_g1_csv_to_proto.py \
    --input-dir data/bones_seed_g1/source \
    --output-dir data/bones_seed_g1/proto/ \
    --input-fps 120 \
    --output-fps 30 \
    --pos-units m \
    --rot-format quat_wxyz \
    --joint-units deg \
    --no-has-frame-column \
    --force-remake

# Test Deployment in Mujoco and convert to 50 FPS (Change input filename)
python deployment/test_tracker_mujoco.py \
    --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --motion data/kimodo_output_proto/g1_2steps.motion \
    --cache-motion --render

# Package (only for training, not needed for deployment)
python protomotions/components/motion_lib.py \
    --motion-path data/kimodo_output_proto/ \
    --output-file data/kimodo_output_proto/g1_3steps.pt

python protomotions/components/motion_lib.py \
    --motion-path data/bones_seed_g1/proto/ \
    --output-file data/bones_seed_g1/g1_bones_seed_selection.pt

# Visualize
python examples/motion_libs_visualizer.py \
    --motion_files data/g1-kimodo-generated/kimodo_g1_motions.pt \
    --robot g1 \
    --simulator isaaclab

# Unpack pt files
scripts/extract_motion_files_from_packaged_motionlib.py data/kimodo_pt/kimodo_reach_v11v10.pt --output-dir data/kimodo_pt