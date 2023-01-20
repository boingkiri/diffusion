# nohup python3 run_cifar10.py &
# tail -f nohup.out

nohup python3 main.py \
    --config configs/ddpm.yml \
    --do_sampling \
    --model ddpm \
    --num_sampling 50000 \
    &
tail -f nohup.out

