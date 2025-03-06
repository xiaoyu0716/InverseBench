# python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg problem.data.id_list=0-4 problem.model.sigma_noise=1.0 exp_name=central-sigma1
# python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg problem.data.id_list=0-4 problem.model.sigma_noise=2.0 exp_name=central-sigma2
# python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg problem.data.id_list=0-4 problem.model.sigma_noise=1.0 exp_name=forward-sigma1 algorithm.method.is_central=False
# python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg problem.data.id_list=0-4 problem.model.sigma_noise=2.0 exp_name=forward-sigma2 algorithm.method.is_central=False
# python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg problem.data.id_list=0-4 exp_name=forward-sigma0 algorithm.method.is_central=False
# python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg problem.data.id_list=0-4 exp_name=central-sigma0 problem.model.adaptive=False
# CUDA_VISIBLE_DEVICES=0 python main.py pretrain=navier-stokes problem=navier-stokes_ds4 algorithm=dps_gsg exp_name=central-sigma0 problem.model.adaptive=False algorithm.method.is_central=True algorithm.method.guidance_scale=0.1
# CUDA_VISIBLE_DEVICES=0 python main.py pretrain=navier-stokes problem=navier-stokes_ds4 algorithm=dps_gsg exp_name=central-sigma1 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
# CUDA_VISIBLE_DEVICES=0 python main.py pretrain=navier-stokes problem=navier-stokes_ds4 algorithm=dps_gsg exp_name=central-sigma2 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1

# CUDA_VISIBLE_DEVICES=6 python main.py pretrain=navier-stokes problem=navier-stokes_ds4 algorithm=dps_gsg exp_name=forward-sigma0 problem.model.adaptive=False algorithm.method.is_central=False algorithm.method.guidance_scale=0.1
# CUDA_VISIBLE_DEVICES=6 python main.py pretrain=navier-stokes problem=navier-stokes_ds4 algorithm=dps_gsg exp_name=forward-sigma1 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
# CUDA_VISIBLE_DEVICES=6 python main.py pretrain=navier-stokes problem=navier-stokes_ds4 algorithm=dps_gsg exp_name=forward-sigma2 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1

# CUDA_VISIBLE_DEVICES=5 python main.py pretrain=navier-stokes problem=navier-stokes_ds8 algorithm=dps_gsg exp_name=forward-sigma0 problem.model.adaptive=False algorithm.method.is_central=False algorithm.method.guidance_scale=0.1
# CUDA_VISIBLE_DEVICES=5 python main.py pretrain=navier-stokes problem=navier-stokes_ds8 algorithm=dps_gsg exp_name=forward-sigma1 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
# CUDA_VISIBLE_DEVICES=5 python main.py pretrain=navier-stokes problem=navier-stokes_ds8 algorithm=dps_gsg exp_name=forward-sigma2 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1

CUDA_VISIBLE_DEVICES=3 python main.py pretrain=navier-stokes problem=navier-stokes_ds8 algorithm=dps_gsg exp_name=forward-sigma0 problem.model.adaptive=False algorithm.method.is_central=True algorithm.method.guidance_scale=0.1
CUDA_VISIBLE_DEVICES=3 python main.py pretrain=navier-stokes problem=navier-stokes_ds8 algorithm=dps_gsg exp_name=forward-sigma1 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
CUDA_VISIBLE_DEVICES=3 python main.py pretrain=navier-stokes problem=navier-stokes_ds8 algorithm=dps_gsg exp_name=forward-sigma2 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1
