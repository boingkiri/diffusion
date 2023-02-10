# nohup python3 run_cifar10.py &
# tail -f nohup.out

# nohup python3 main.py \
#     --config config.yml \
#     --do_train \
#     --do_sampling \
#     --num_sampling 50000 \
#     &

# tail -f nohup.out

python3 main.py \
    --config-path configs \
    --config-name config \
    --do_train \
    --do_sampling \
    --num_sampling 50000 

