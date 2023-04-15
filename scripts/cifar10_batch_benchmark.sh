run1="deepspeed --num_gpus 1 train.py --deepspeed --deepspeed_config ./configs/base_config.json     --gpus 1"
run4="deepspeed --num_gpus 4 train.py --deepspeed --deepspeed_config ./configs/base_config.json     --gpus 4"


# ${run1} --model darts --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 320 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 1024 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 1024 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 2048 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 4096 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 5120 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model darts --img_size 32 --batch_size 5120 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run4} --model darts --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 320 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 1024 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 1024 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 2048 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 4096 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 5120 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model darts --img_size 32 --batch_size 5120 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run1} --model vit_l --img_size 32 --batch_size 2560 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 4800 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 4800 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 5600 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 2560 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 4200 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 4600 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_l --img_size 32 --batch_size 6400 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run4} --model vit_l --img_size 32 --batch_size 4200 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 4200 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 2560 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 4200 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 5600 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 5600 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 2560 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 4200 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 6400 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_l --img_size 32 --batch_size 6400 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run1} --model vit_h --img_size 32 --batch_size 24 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 48 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 128 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 24 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 48 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run4} --model vit_h --img_size 32 --batch_size 24 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 48 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 128 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 24 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 48 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_h --img_size 32 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# # ${run1} --model resnet152 --img_size 32 --batch_size 128 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 256 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 256 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# # ${run1} --model resnet152 --img_size 32 --batch_size 256 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run1} --model vit_g --img_size 32 --batch_size 4 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 1536 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 4 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model vit_g --img_size 32 --batch_size 1536 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run4} --model vit_g --img_size 32 --batch_size 4 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 1536 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 4 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model vit_g --img_size 32 --batch_size 1536 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
${run4} --model vit_g --img_size 32 --batch_size 4800 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
${run4} --model vit_g --img_size 32 --batch_size 4800 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 

# ${run1} --model mobilenet --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 128 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 98 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 160 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 320 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run1} --model mobilenet --img_size 32 --batch_size 320 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 

# ${run4} --model mobilenet --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 128 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 256 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0
# ${run4} --model mobilenet --img_size 32 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 1 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0  
# ${run4} --model mobilenet --img_size 32 --batch_size 98 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 160 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 0     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 320 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload null --seed 666 --exp_name null --debug 0 
# ${run4} --model mobilenet --img_size 32 --batch_size 320 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --num_stages -1 --use_zero 1     --zero_stage 3 --offload cpu --seed 666 --exp_name null --debug 0 
