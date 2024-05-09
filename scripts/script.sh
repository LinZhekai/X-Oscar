TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=1 python -m apps.run_XOscar \
    --config configs/XOscar.yaml \
    --name XOscar_Spiderman --text Spiderman