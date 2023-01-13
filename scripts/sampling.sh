nohup python3 main.py \
    --config config.yml \
    --do_sampling \
    --num_sampling 50000 \
    &
tail -f nohup.out


# python3 main.py \
#     --config config.yml \
#     --do_sampling\
#     --num_sampling 50000