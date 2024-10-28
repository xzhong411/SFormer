
# ###### Generating CAMs ##########
# CUDA_VISIBLE_DEVICES=0 python main.py  --data-set VOC12MS \
#                 --img-list Dataset/VOC2012/ImageSets/Segmentation \
#                 --data-path Dataset/VOC2012 \
#                 --gen_attention_maps \
#                 --resume saved_model/checkpoint.pth \
#                 --cam-npy-dir results/voc/cam-npy \


# # # # ######### Evaluating the generated CAMs   ##########

python evaluation.py --list voc12/train_id.txt \
                     --gt_dir Dataset/VOC2012/SegmentationClass \
                     --logfile saved_model/cam.txt \
                     --type npy \
                     --curve True \
                     --predict_dir results/voc/cam-npy \
                     --comment "tr"
