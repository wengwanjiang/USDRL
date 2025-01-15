gpu=0,1 #$1
dir=ntu60_xs_j_dste #$2

mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0005   --batch-size 356  --schedule 351 --epochs 451 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone DSTE \
 --pre-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_pretrain.log

dir=ntu60_xs_j_sttr
mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0005   --batch-size 356  --schedule 351 --epochs 451 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone STTR \
 --pre-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_pretrain.log

exit 0


# ------------------------------------------------------------------------------------------------------------- #
# For bone and motion modality, just modify the --moda argument to bone or motion
dir=ntu60_xv_j_dste
mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0005   --batch-size 356  --schedule 351 --epochs 451 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone DSTE \
 --pre-dataset ntu60 --protocol cross_view | tee -a ./checkpoint/${dir}/${dir}_pretrain.log

dir=ntu120_xs_j_dste
mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0005   --batch-size 356  --schedule 351 --epochs 451 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone DSTE \
 --pre-dataset ntu120 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_pretrain.log

dir=ntu120_xe_j_dste
mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0005   --batch-size 356  --schedule 351 --epochs 451 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone DSTE \
 --pre-dataset ntu120 --protocol cross_setup | tee -a ./checkpoint/${dir}/${dir}_pretrain.log

dir=v2_xs_j_dste
mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0005   --batch-size 512 --schedule 801 --epochs 1201 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone DSTE \
 --pre-dataset pku_v2 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_pretrain.log