# nohup python3 run_cifar10.py &
# tail -f nohup.out

nohup python3 main.py --config config.yml --do_train true &
tail -f nohup.out