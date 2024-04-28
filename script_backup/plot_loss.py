import matplotlib.pyplot as plt
import re

# Initialize a list to store the loss values
loss_values = []

# Open the text file and read lines
with open('selena_adam30-50error.txt', 'r') as file:  # Replace 'your_file.txt' with your file path
    for line in file:
        # Use regular expression to find loss values
        match = re.search(r'loss=([0-9.]+)', line)
        if match:
            # Convert the extracted value to float and append to the list
            loss_values.append(float(match.group(1)))

# Plotting
plt.plot(loss_values, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.legend()
plt.savefig("Loss_logSelena_adam_30-50epoch.png") 
