*Special Thanks to "https://github.com/abdulfatir/prototypical-networks-tensorflow"

# Diverse Few-Shot Text Classification with Multiple Metrics, Yu et al, 2018
Configurations in utils.args.py

## Requirements
tensorflow = 2.2
cvxpy = 1.1

/#TODO : put things in an organized fashion(More capsulation needed)

## How to Run
1. python data_util.py -> Create : "vocab.pkl", and 'data_id'
2. python build_matrix.py -> Create : 'cluster.json', 'ckpt_log' , 'csv_log'
3. python FSL.py -> Create : 'ckpt_log_fsl','alpha'


#### NOTE)
Hardware info : VRAM 8GB GPU


Allow growth(Dynamic VRAM allocation) should be on, use about 3GB VRAM.


Per-task classifier train took about 10~11 min, while transfer(60% measure) took 105 min.


Transfer performance matrix took more than a hour(with max 3 epochs), but better pipeline design would enhance the computation speed(GPU util too low).



