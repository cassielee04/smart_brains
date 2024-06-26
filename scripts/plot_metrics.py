"""
this scripts plots dice score and F1 score after the metrics.csv file is generated by the metrics.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MAIN_PATH = os.getenv('MODEL_PATH', './default_model_path')
print("MAIN_PATH",MAIN_PATH)
FILE_PATH = f"{MAIN_PATH}/metrics.csv"
# SCORE_TYPE = "F1" # ["F1", "DICE"]
# CLASS_TYPE = "1:NecroticCore" # ['0:NoTumor', '1:NecroticCore', '2:Edema','3:Enhancing']
 
data = pd.read_csv(FILE_PATH)
print(f"data.columns = {data.columns}") # 'Model Name', 'Class Name', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'Dice']
print(f"data['Model Name'].unique() = {data['Model Name'].unique()}") # ['expAugmented_flair' 'expAugmented_t1ce' 'expAugmented_t1ce-flair' 'exp_flair' 'exp_t1ce' 'exp_t1ce-flair']
print(f"data['Class Name'].unique() = {data['Class Name'].unique()}") # ['0:NoTumor' '1:NecroticCore' '2:Edema' '3:Enhancing']


def plot_score(data, score_type, class_type, main_path=MAIN_PATH):
    score_data = data[['Model Name', 'Class Name', score_type]].copy()
    score_data['Augmented'] = score_data['Model Name'].str.contains('Augmented')
    score_data['Augmented'] = score_data['Augmented'].map({True: 'Augmented', False: 'Not Augmented'})
    score_data["Modal Type"] = score_data['Model Name'].apply(lambda x: x.split("_")[-1])
    
    # Sorting 'Augmented' column for consistent plot order
    augmented_order = ['Not Augmented', 'Augmented']
    
    selected_data = score_data[score_data['Class Name'] == class_type].copy()
    
    # Clear any existing plots
    plt.clf()
    plt.figure(figsize=(10, 6))
    
    sns.barplot(data=selected_data, 
                x='Modal Type', 
                y=score_type, 
                hue='Augmented', 
                hue_order=augmented_order, 
                # palette='deep', 
                dodge=True)
    plt.title(f'{score_type} Scores for {class_type} based on Modal Type',fontsize=20)
    plt.ylabel(f'{score_type} Score',fontsize=18)
    plt.xlabel('Model Type',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.ylim(0.0, 1.1) 
    
    # Save the figure
    plt.savefig(f"{main_path}/plot_{score_type}_{class_type}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free up system resources

from PIL import Image

def combine_images(file_paths, output_path):
    images = [Image.open(file) for file in file_paths]
    
    # Assuming all images are the same size, get dimensions of the first image
    scale_ratio = 1.1
    width, height = (int(images[0].size[0]*scale_ratio), int(images[0].size[1]*scale_ratio))
    print(f"width = {width}")
    print(f"width = {height}")
    
    # Create a new image with a white background
    new_image = Image.new('RGB', (2 * width, 2 * height), 'white')
    
    # Paste images into the new image
    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (width, 0))
    new_image.paste(images[2], (0, height))
    new_image.paste(images[3], (width, height))
    
    # Save the new image
    new_image.save(output_path)


class_types = ['0:NoTumor', '1:NecroticCore', '2:Edema','3:Enhancing']
score_types = ["F1", "Dice"]
file_names = []
for score_type in score_types:
    file_names_each = []
    for class_type in class_types:
        print(f"score_type = {score_type}")
        print(f"class_type = {class_type}")
        plot_score(data, score_type, class_type)
        file_name = f"{MAIN_PATH}/plot_{score_type}_{class_type}.png"
        file_names_each.append(file_name)
    file_names.append(file_names_each)
# print(file_names)
    
# Example usage:
for i in range(len(score_types)):
    file_names_each = file_names[i]
    score_type = score_types[i]
    combine_images(file_names_each, f'{MAIN_PATH}/plot_{score_type}_combined.png')




