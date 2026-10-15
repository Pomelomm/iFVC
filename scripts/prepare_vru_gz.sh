#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  echo "Usage: bash scripts/prepare_vru_gz.sh {all|extract|colmap|weights|config} --data-root PATH --output-root PATH"
  exit 1
fi
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFVC_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${IFVC_DATA_ROOT:-}"
OUTPUT_ROOT="${IFVC_OUTPUT_ROOT:-}"
SCENE_DIR="${VRU_SCENE_DIR:-}"
VIDEO_DIR="${VRU_VIDEO_DIR:-}"
CHECKPOINT_DIR="${TORCH_CHECKPOINT_DIR:-${HOME}/.cache/torch/hub/checkpoints}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --scene-dir)
      SCENE_DIR="$2"
      shift 2
      ;;
    --video-dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$SCENE_DIR" && -n "$DATA_ROOT" ]]; then
  SCENE_DIR="$DATA_ROOT/prepared/VRU/Dataset/VRU_gz"
fi
if [[ -z "$VIDEO_DIR" && -n "$DATA_ROOT" ]]; then
  VIDEO_DIR="$DATA_ROOT/raw/VRU_Basketball/gz/videos"
fi

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing $name"
    exit 1
  fi
}

extract_frames() {
  require_value "--data-root or --video-dir" "$VIDEO_DIR"
  require_value "--data-root or --scene-dir" "$SCENE_DIR"

  local -a videos
  local extract_tmp video camera_id camera_name camera_complete
  local frame_id frame_name frame_image_dir

  mkdir -p "$SCENE_DIR"
  mapfile -t videos < <(find "$VIDEO_DIR" -type f -name '*.mp4' -print | sort -V)
  if [[ "${#videos[@]}" -eq 0 ]]; then
    echo "No MP4 files found in $VIDEO_DIR"
    exit 1
  fi

  extract_tmp=$(mktemp -d "$SCENE_DIR/.extract.XXXXXX")
  trap 'rm -rf "$extract_tmp"' RETURN

  for video in "${videos[@]}"; do
    camera_id=$(basename "$video" .mp4)
    printf -v camera_name 'cam%02d.png' "$((10#$camera_id))"
    camera_complete=true

    for frame_id in $(seq 0 249); do
      printf -v frame_name 'frame%06d' "$frame_id"
      if [[ "$frame_id" -eq 0 ]]; then
        frame_image_dir="$SCENE_DIR/$frame_name/inputs"
      else
        frame_image_dir="$SCENE_DIR/$frame_name"
      fi
      if [[ ! -f "$frame_image_dir/$camera_name" ]]; then
        camera_complete=false
        break
      fi
    done

    if [[ "$camera_complete" == true ]]; then
      continue
    fi

    mkdir -p "$extract_tmp/$camera_id"
    ffmpeg -hide_banner -loglevel error -i "$video" \
      -vf "select='between(n,0,249)'" -vsync 0 -frames:v 250 -start_number 0 \
      "$extract_tmp/$camera_id/frame%06d.png"

    for frame_id in $(seq 0 249); do
      printf -v frame_name 'frame%06d' "$frame_id"
      if [[ "$frame_id" -eq 0 ]]; then
        frame_image_dir="$SCENE_DIR/$frame_name/inputs"
      else
        frame_image_dir="$SCENE_DIR/$frame_name"
      fi
      mkdir -p "$frame_image_dir"
      if [[ ! -f "$frame_image_dir/$camera_name" ]]; then
        mv "$extract_tmp/$camera_id/$frame_name.png" \
          "$frame_image_dir/$camera_name"
      fi
    done
    rm -rf "$extract_tmp/$camera_id"
  done

  rm -rf "$extract_tmp"
  trap - RETURN
}

prepare_colmap() {
  require_value "--data-root or --scene-dir" "$SCENE_DIR"

  local frame0="$SCENE_DIR/frame000000"
  local frame_id frame_name frame_dir file
  mkdir -p "$frame0/distorted/sparse"

  if [[ ! -f "$frame0/sparse/0/cameras.bin" ]]; then
    colmap feature_extractor \
      --database_path "$frame0/distorted/database.db" \
      --image_path "$frame0/inputs" \
      --ImageReader.single_camera 1 \
      --ImageReader.camera_model OPENCV \
      --SiftExtraction.use_gpu 0

    colmap exhaustive_matcher \
      --database_path "$frame0/distorted/database.db" \
      --SiftMatching.use_gpu 0

    colmap mapper \
      --database_path "$frame0/distorted/database.db" \
      --image_path "$frame0/inputs" \
      --output_path "$frame0/distorted/sparse"

    colmap image_undistorter \
      --image_path "$frame0/inputs" \
      --input_path "$frame0/distorted/sparse/0" \
      --output_path "$frame0" \
      --output_type COLMAP

    mkdir -p "$frame0/sparse/0"
    for file in cameras.bin images.bin points3D.bin; do
      if [[ -f "$frame0/sparse/$file" ]]; then
        mv "$frame0/sparse/$file" "$frame0/sparse/0/$file"
      fi
    done
  fi

  for frame_id in $(seq 1 249); do
    printf -v frame_name 'frame%06d' "$frame_id"
    frame_dir="$SCENE_DIR/$frame_name"

    if [[ -f "$frame_dir/sparse/0/cameras.bin" && -d "$frame_dir/images" ]]; then
      continue
    fi

    colmap image_undistorter \
      --image_path "$frame_dir" \
      --input_path "$frame0/distorted/sparse/0" \
      --output_path "$frame_dir" \
      --output_type COLMAP

    mkdir -p "$frame_dir/sparse/0"
    for file in cameras.bin images.bin points3D.bin; do
      if [[ -f "$frame_dir/sparse/$file" ]]; then
        mv "$frame_dir/sparse/$file" "$frame_dir/sparse/0/$file"
      fi
    done
  done
}

prepare_weights() {
  mkdir -p "$CHECKPOINT_DIR"

  if [[ ! -f "$CHECKPOINT_DIR/vgg16-397923af.pth" ]]; then
    aria2c --continue=true --max-connection-per-server=16 --split=16 \
      --min-split-size=1M --file-allocation=none \
      --dir="$CHECKPOINT_DIR" \
      --out='vgg16-397923af.pth' \
      'https://hf-mirror.com/ByteDance/LatentSync-1.5/resolve/main/auxiliary/vgg16-397923af.pth'
  fi

  if [[ ! -f "$CHECKPOINT_DIR/vgg.pth" ]]; then
    curl -fL --retry 5 \
      -o "$CHECKPOINT_DIR/vgg.pth" \
      'https://ghfast.top/https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth'
  fi
}

prepare_config() {
  require_value "--data-root" "$DATA_ROOT"
  require_value "--output-root" "$OUTPUT_ROOT"

  cd "$IFVC_ROOT"
  mkdir -p .cache/configs

  python - "$DATA_ROOT" "$OUTPUT_ROOT" <<'PY'
import json
import os
import sys

data_root, output_root = sys.argv[1:]
source = "configs/VRU/VRU_gz/VRU_gz_0.004_1.0.json"
with open(source, "r") as f:
    config = json.load(f)

config["base_source_path"] = os.path.join(data_root, "prepared", "VRU", "Dataset")
config["base_model_path"] = output_root
config["frame_start"] = 0
config["frame_end"] = 249
config["Init_frame_params"]["iterations"] = 30000
config["Init_frame_params"]["test_iterations"] = [30000]
config["Init_frame_params"]["save_iterations"] = [30000]
config["I_frame_params"]["iterations"] = 30000
config["I_frame_params"]["test_iterations"] = [30000]
config["I_frame_params"]["save_iterations"] = [30000]
config["P_frame_params"]["iterations"] = 500
config["P_frame_params"]["first_ref_iteration"] = 30000
config["P_frame_params"]["ref_iter"] = 500
config["P_frame_params"]["test_iterations"] = [500]
config["P_frame_params"]["save_iterations"] = [500]

with open(".cache/configs/VRU_gz_full.json", "w") as f:
    json.dump(config, f, indent=2)
PY
}

case "$ACTION" in
  all)
    extract_frames
    prepare_colmap
    prepare_weights
    prepare_config
    ;;
  extract)
    extract_frames
    ;;
  colmap)
    prepare_colmap
    ;;
  weights)
    prepare_weights
    ;;
  config)
    prepare_config
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Use one of: all, extract, colmap, weights, config"
    exit 1
    ;;
esac
