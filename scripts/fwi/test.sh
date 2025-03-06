# Adam
python main.py pretrain=fwi problem=fwi algorithm=adam problem.data.id_list=1-10 exp_name=constant
# LBFGS
python main.py pretrain=fwi problem=fwi algorithm=lbfgs problem.data.id_list=1-10 exp_name=constant
# DiffPIR
python main.py problem=fwi pretrain=fwi algorithm=diffpir algorithm.method.xi=0.11 algorithm.method.lamb=80.6 algorithm.method.sigma_n=0.28 algorithm.method.diffusion_scheduler_config.num_steps=600

# DPS
python main.py pretrain=fwi problem=fwi algorithm=dps algorithm.method.guidance_scale=0.01 exp_name=rep0 problem.data.id_list=1-10