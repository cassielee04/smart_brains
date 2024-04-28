# This is designed to take log files to plot training and validation learning curves

import pandas as pd
import matplotlib.pyplot as plt

def update_scores_from_files(file_paths):
    # Initialize an empty DataFrame to store the scores
    df_scores = pd.DataFrame(columns=['epoch', 'training_score', 'validation_score'])
    
    # Process each file sequentially
    for file_path in file_paths:
        temp_scores = {'epoch': [], 'training_score': [], 'validation_score': []}
        current_epoch = 0
        
        with open(file_path, 'r') as file:
            for line in file:
                # Update the epoch number if a new epoch starts
                if 'Epoch:' in line:
                    parts = line.split('Epoch: ')
                    current_epoch = int(parts[1])
                    
                # Check if the line contains the training dice score
                if 'Training Average Dice Score:' in line:
                    parts = line.split('Training Average Dice Score: ')
                    training_score = float(parts[1])
                    temp_scores['training_score'].append(training_score)
                    temp_scores['epoch'].append(current_epoch)
                
                # Check if the line contains the validation dice score
                if 'Validation Average Dice Score:' in line:
                    parts = line.split('Validation Average Dice Score: ')
                    validation_score = float(parts[1])
                    temp_scores['validation_score'].append(validation_score)
        
        # Create a temporary DataFrame from the extracted scores
        temp_df = pd.DataFrame(temp_scores)
        
        # Merge the new data into the main DataFrame, updating existing entries
        if df_scores.empty:
            df_scores = temp_df
        else:
            df_scores = df_scores.merge(temp_df, on='epoch', how='outer', suffixes=('', '_new'))
        
            # Update existing entries with new data if available
            for col in ['training_score', 'validation_score']:
                df_scores[col] = df_scores[col + '_new'].combine_first(df_scores[col])
                df_scores.drop(columns=[col + '_new'], inplace=True)

    return df_scores

def plot_and_save_dice_scores(df, modal_type, filename):
    # Plotting the training and validation scores
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['training_score'], label='Training Score', marker='o')
    plt.plot(df['epoch'], df['validation_score'], label='Validation Score', marker='x')
    plt.title(f'{modal_type.capitalize()} Training and Validation Scores per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the figure to a file
    plt.close()  # Close the figure window to free up memory

# List of file paths
MAIN_PATH = "/projectnb/ds598/projects/smart_brains"

modal_type = "t1ce"
file_paths = [f"{MAIN_PATH}/yuke_t1ce_opAdam_lr0.0001_bs16_epoch0_200_training.log"]
# Update and create the DataFrame with scores
df_scores = update_scores_from_files(file_paths)
df_scores.to_csv(f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores_new5.csv', index=False)
plot_and_save_dice_scores(df_scores, modal_type, f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores_new5.png')

# ### Flair
# modal_type = "flair"
# flair_file_paths = [
#     f'{MAIN_PATH}/Selena_Flair_opAdam_lr0.0001_bs16_epoch0_20_training.log',
#     f'{MAIN_PATH}/Selena_Flair_opAdam_lr1e-06_bs16_epoch20_40_training.log',
#     f'{MAIN_PATH}/Selena_Flair_opAdam_lr1e-08_bs16_epoch33_83_training.log'
# ]
# # Update and create the DataFrame with scores
# df_scores = update_scores_from_files(flair_file_paths)
# df_scores.to_csv(f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores.csv', index=False)
# plot_and_save_dice_scores(df_scores, modal_type, f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores.png')


# ### T1CE
# modal_type = "t1ce"
# t1ce_file_paths = [
#     f'{MAIN_PATH}/Selena_T1CE_opAdam_lr0.0001_bs16_epoch0_20_training.log',
#     f'{MAIN_PATH}/Selena_T1CE_opAdam_lr1e-06_bs16_epoch19_39_training.log',
#     f'{MAIN_PATH}/Selena_T1CE_opAdam_lr1e-08_bs16_epoch35_85_training.log'
# ]
# # Update and create the DataFrame with scores
# df_scores = update_scores_from_files(t1ce_file_paths)
# df_scores.to_csv(f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores.csv', index=False)
# plot_and_save_dice_scores(df_scores, modal_type, f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores.png')



# ### Flair+T1CE
# modal_type = "flair_t1ce_combined"
# flair_t1ce_file_paths = [
#     f'{MAIN_PATH}/Selena_Flair-T1ce_opAdam_lr0.0001_bs16_epoch0_20_training.log',
#     f'{MAIN_PATH}/Selena_Flair-T1ce_opAdam_lr1e-06_bs16_epoch13_33_training.log',
#     f'{MAIN_PATH}/Selena_Flair-T1ce_opAdam_lr1e-08_bs16_epoch31_81_training.log'
# ]
# # Update and create the DataFrame with scores
# df_scores = update_scores_from_files(flair_t1ce_file_paths)
# df_scores.to_csv(f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores.csv', index=False)
# plot_and_save_dice_scores(df_scores, modal_type, f'{MAIN_PATH}/{modal_type}_training_validation_dice_scores.png')
