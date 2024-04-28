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
    plt.title(f'{modal_type.capitalize()} Training and Validation Dice Scores per Epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Dice Score', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.ylim(0.0, 1.0) 
    plt.savefig(filename)  # Save the figure to a file
    plt.close()  # Close the figure window to free up memory

def process_DiceLog_plot(save_path, modal_type, EXP_Name, log_file_paths):
    """
    Process training logs, create a DataFrame with scores, and save the results as a CSV and a plot.
    
    Parameters:
        save_path (str): The directory path where generated files are stored.
        modal_type (str): The type of modality, e.g., 'flair', 't1ce', etc.
        log_file_paths (str): The path to log file of the model
    
    Returns:
        None
    """
   
    # Update and create the DataFrame with scores
    df_scores = update_scores_from_files(log_file_paths)
    csv_path = f'{save_path}/{EXP_Name}_training_validation_dice_scores.csv'
    df_scores.to_csv(csv_path, index=False)
    
    # Plot and save the dice scores
    plot_path = f'{save_path}/{EXP_Name}_training_validation_dice_scores.png'
    plot_and_save_dice_scores(df_scores, modal_type, plot_path)

    print(f"Scores and plots have been saved to {csv_path} and {plot_path}")


def main():
    EXP_Name = "exp_flair_opAdam_lr0.0001_bs16_epoch0_200"
    save_path = f"/projectnb/ds598/projects/smart_brains/RESULTS/{EXP_Name}"
    modal_type = EXP_Name.split("_")[1]
    log_file_paths = f"/projectnb/ds598/projects/smart_brains/{EXP_Name}_training.log"
    process_DiceLog_plot(save_path, modal_type, EXP_Name, [log_file_paths])

    EXP_Name = "exp_t1ce_opAdam_lr0.0001_bs16_epoch0_200"
    save_path = f"/projectnb/ds598/projects/smart_brains/RESULTS/{EXP_Name}"
    modal_type = EXP_Name.split("_")[1]
    log_file_paths = f"/projectnb/ds598/projects/smart_brains/{EXP_Name}_training.log"
    process_DiceLog_plot(save_path, modal_type, EXP_Name, [log_file_paths])

    EXP_Name = "exp_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200"
    save_path = f"/projectnb/ds598/projects/smart_brains/RESULTS/{EXP_Name}"
    modal_type = EXP_Name.split("_")[1]
    log_file_paths = f"/projectnb/ds598/projects/smart_brains/{EXP_Name}_training.log"
    process_DiceLog_plot(save_path, modal_type, EXP_Name, [log_file_paths])

    EXP_Name = "expAugmented_flair_opAdam_lr0.0001_bs16_epoch0_200"
    save_path = f"/projectnb/ds598/projects/smart_brains/RESULTS/{EXP_Name}"
    modal_type = EXP_Name.split("_")[1]
    log_file_paths = f"/projectnb/ds598/projects/smart_brains/{EXP_Name}_training.log"
    process_DiceLog_plot(save_path, modal_type, EXP_Name, [log_file_paths])

    EXP_Name = "expAugmented_t1ce_opAdam_lr0.0001_bs16_epoch0_200"
    save_path = f"/projectnb/ds598/projects/smart_brains/RESULTS/{EXP_Name}"
    modal_type = EXP_Name.split("_")[1]
    log_file_paths = f"/projectnb/ds598/projects/smart_brains/{EXP_Name}_training.log"
    process_DiceLog_plot(save_path, modal_type, EXP_Name, [log_file_paths])

    EXP_Name = "expAugmented_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200"
    save_path = f"/projectnb/ds598/projects/smart_brains/RESULTS/{EXP_Name}"
    modal_type = EXP_Name.split("_")[1]
    log_file_paths = f"/projectnb/ds598/projects/smart_brains/{EXP_Name}_training.log"
    process_DiceLog_plot(save_path, modal_type, EXP_Name, [log_file_paths])


main()
