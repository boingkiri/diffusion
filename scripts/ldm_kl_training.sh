# nohup python3 run_cifar10.py &
# tail -f nohup.out

nohup python3 main.py \
    --config configs/ldm_kl.yml \
    --do_train \
    --do_sampling \
    --model ldm \
    --num_sampling 50000 \
    &
tail -f nohup.out

