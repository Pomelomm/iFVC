scenes=("discussion" "trimming" "vrheadset")
for scene in ${scenes[@]}; do
    for i in {0..10}; do
        CUDA_VISIBLE_DEVICES=3 python train_videos.py --config_path ./configs/${scene}/${scene}_0.001_1.0.json --init_name _${i}
        CUDA_VISIBLE_DEVICES=3 python train_videos.py --config_path ./configs/${scene}/${scene}_0.002_1.0.json --init_name _${i}
        CUDA_VISIBLE_DEVICES=3 python train_videos.py --config_path ./configs/${scene}/${scene}_0.003_1.0.json --init_name _${i}
        CUDA_VISIBLE_DEVICES=3 python train_videos.py --config_path ./configs/${scene}/${scene}_0.004_1.0.json --init_name _${i}
    done
done