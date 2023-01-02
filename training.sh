# nohup python3 run_cifar10.py &
# tail -f nohup.out

nohup python3 main.py \
    --config config.yml \
    --do_train \
    --do_sampling \
    &
tail -f nohup.out