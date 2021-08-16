VideoName='FP006542HD02'
Attribute='Smiling'
Scale='1'
Sigma='3' # Choose appropriate gaussian filter size
VideoDir='./data/video'
Path=${PWD}


# Cut video to frames
python video_processing.py --function 'video_to_frames' --video_path ${VideoDir}/${VideoName}.mp4 #--resize

# Crop and align the faces in each frame
python video_processing.py --function 'align_frames' --video_path ${VideoDir}/${VideoName}.mp4 --filter_size=${Sigma} --optical_flow

# Project each frame to StyleGAN2 latent space
cd pixel2style2pixel/
python scripts/inference.py --checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=${Path}/outputs/video/${VideoName}/${VideoName}_crop_align \
--exp_dir=${Path}/outputs/video/${VideoName}/${VideoName}_crop_align_latent \
--test_batch_size=1

# Achieve latent manipulation
cd ${Path}
python video_processing.py --function 'latent_manipulation' --video_path ${VideoDir}/${VideoName}.mp4 --attr ${Attribute} --alpha=${Scale}

# Reproject the manipulated frames to the original video
python video_processing.py --function 'reproject_origin' --video_path ${VideoDir}/${VideoName}.mp4 --seamless
python video_processing.py --function 'reproject_manipulate' --video_path ${VideoDir}/${VideoName}.mp4 --attr ${Attribute} --seamless
python video_processing.py --function 'compare_frames' --video_path ${VideoDir}/${VideoName}.mp4 --attr ${Attribute} --strs 'Original,Projected,Manipulated'
