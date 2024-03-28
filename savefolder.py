import os
expName = "cassie"
save_image_path = f"/projectnb/ds598/projects/smart_brains/saved_images_{expName}"
os.makedirs(save_image_path,exist_ok=True)
is_created = os.path.exists(save_image_path)
print(is_created)
#save_prediction_as_imgs(epoch, train_loader, model,folder=save_image_path, device=DEVICE)

   