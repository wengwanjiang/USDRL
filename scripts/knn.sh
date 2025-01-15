gpu=0 #$1
dir=ntu60_xs_j_dste #$2
node=45 #$3
CUDA_VISIBLE_DEVICES="${gpu}" python action_retrieval.py \
  --lr 0.006 --batch-size 512 --backbone DSTE --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_recog.log

dir=ntu60_xs_j_sttr
CUDA_VISIBLE_DEVICES="${gpu}" python action_retrieval.py \
  --lr 0.03 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_recog.log
  
# For other dataset and protocols, just modfiy the --finetune-dataset and --protocol arguments