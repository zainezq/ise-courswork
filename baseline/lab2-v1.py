import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 5  # More repetitions for stability
    train_frac = 0.7
    random_seed = 42  # Fixed for reproducibility
    output_file = "output_proposed.txt"

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    with open(output_file, "w") as f:
        for current_system in systems:
            datasets_location = f'datasets/{current_system}'
            csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

            for csv_file in csv_files:
                print(f'\n> System: {current_system}, Dataset: {csv_file}')
                f.write(f'\n> System: {current_system}, Dataset: {csv_file}\n')

                data = pd.read_csv(os.path.join(datasets_location, csv_file))

                metrics = {'MAPE': [], 'MAE': [], 'RMSE': [], 'R2': []}

                for current_repeat in range(num_repeats):
                    train_data, test_data = train_test_split(data, train_size=train_frac, random_state=random_seed + current_repeat)
                    training_X, training_Y = train_data.iloc[:, :-1], train_data.iloc[:, -1]
                    testing_X, testing_Y = test_data.iloc[:, :-1], test_data.iloc[:, -1]

                    # Standardize features
                    scaler = StandardScaler()
                    training_X = scaler.fit_transform(training_X)
                    testing_X = scaler.transform(testing_X)

                    # Using GridSearchCV for hyperparameter tuning
                    base_model = GradientBoostingRegressor(random_state=random_seed)
                    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                    grid_search.fit(training_X, training_Y)

                    # Best model from grid search
                    best_model = grid_search.best_estimator_

                    predictions = best_model.predict(testing_X)

                    mape = mean_absolute_percentage_error(testing_Y, predictions)
                    mae = mean_absolute_error(testing_Y, predictions)
                    rmse = np.sqrt(mean_squared_error(testing_Y, predictions))
                    r2 = r2_score(testing_Y, predictions)  # Calculate R² Score

                    metrics['MAPE'].append(mape)
                    metrics['MAE'].append(mae)
                    metrics['RMSE'].append(rmse)
                    metrics['R2'].append(r2)

                avg_mape = np.mean(metrics["MAPE"])
                avg_mae = np.mean(metrics["MAE"])
                avg_rmse = np.mean(metrics["RMSE"])
                avg_r2 = np.mean(metrics["R2"])  # Compute average R² Score

                print(f'Average MAPE: {avg_mape:.2f}')
                print(f'Average MAE: {avg_mae:.2f}')
                print(f'Average RMSE: {avg_rmse:.2f}')
                print(f'Average R²: {avg_r2:.2f}')
                
                f.write(f'Average MAPE: {avg_mape:.2f}\n')
                f.write(f'Average MAE: {avg_mae:.2f}\n')
                f.write(f'Average RMSE: {avg_rmse:.2f}\n')
                f.write(f'Average R²: {avg_r2:.2f}\n')

if __name__ == "__main__":
    main()
