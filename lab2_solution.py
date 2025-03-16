import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible
    """

    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3  # Modify this value to change the number of repetitions
    train_frac = 0.7  # Modify this value to change the training data fraction (e.g., 0.7 for 70%)
    random_seed = 1  # The random seed will be altered for each repeat

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'  # Modify this to specify the location of the datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]  # List all CSV files in the directory

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}')

            data = pd.read_csv(os.path.join(datasets_location, csv_file))  # Load data from CSV file

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': [], 'R2': []}  # Initialize a dict to store results for repeated evaluations

            for current_repeat in range(num_repeats):  # Repeat the process n times
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)  # Change the random seed based on the current repeat
                test_data = data.drop(train_data.index)

                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                model = LinearRegression()  # Initialize a Linear Regression model

                model.fit(training_X, training_Y)  # Train the model with the training data

                predictions = model.predict(testing_X)  # Predict the testing data

                # Calculate evaluation metrics for the current repeat
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))
                r2 = r2_score(testing_Y, predictions)  # Calculate R² Score

                # Store the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)
                metrics['R2'].append(r2)

            # Calculate the average of the metrics for all repeats
            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])
            avg_r2 = np.mean(metrics['R2'])

            print(f'Average MAPE: {avg_mape:.2f}')
            print(f'Average MAE: {avg_mae:.2f}')
            print(f'Average RMSE: {avg_rmse:.2f}')
            print(f'Average R²: {avg_r2:.2f}')


if __name__ == "__main__":
    main()
