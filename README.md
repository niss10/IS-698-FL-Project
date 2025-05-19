# FedCollab: Federated Learning for Personalized Movie Recommendations

## Overview
FedCollab is a final project for the IS 698/800 Federated Learning course, developed by Chaw Maung and Nisarg Patel. The project implements a privacy-preserving movie recommendation system using the MovieLens 1M dataset. It compares centralized machine learning (CML) with federated learning (FL) approaches, incorporating differential privacy (DP) to enhance user data security. The system leverages collaborative filtering with Singular Value Decomposition (SVD) and Multi-Layer Perceptron (MLP) models, achieving near-centralized performance while keeping user data local.

- **Centralized Baseline:** Implements SVD and MLP models using the Surprise and PyTorch libraries.
- **Federated Learning:** Simulates 100 clients using the Flower framework with FedAvg strategy.
- **Privacy Preservation:** Integrates differential privacy with configurable privacy budgets (ε=1.0, ε=5.0).
- **Comprehensive Evaluation:** Measures MSE, RMSE, Precision@10, and Recall@10 to compare CML and FL performance.

## Prerequisites
To run this project, ensure the following are installed:

- Python 3.8+ programming language.
- Google Colab Recommended environment (CPU runtime, 12.7 GB RAM, 107.7 GB disk).

**Libraries:**
- numpy==1.24.3 (for CML and FL)
- pandas==2.2.3
- scikit-learn==1.6.1
- torch==2.7.0 (CPU-only build)
- flwr==1.18.0 (with [simulation] for FL)
- scikit-surprise (for SVD in centralized learning)
- matplotlib (for visualizations)

**Dataset:** [MovieLens 1M dataset (ratings.dat, users.dat, movies.dat) from GroupLens.](https://grouplens.org/datasets/movielens/1m/)

## Installation

- Clone the Repository: git clone https://github.com/your-username/fedcollab.git
- cd fedcollab
- Download the MovieLens 1M Dataset: https://grouplens.org/datasets/movielens/1m/
- Obtain ratings.dat, users.dat, and movies.dat from GroupLens.
- Place these files in the project root directory.

**Set Up Google Colab:**
- Upload the project files to google Colab notebook. or Directly open Google Notebook from link top-left corner of file 
- Ensure the runtime is set to CPU with sufficient RAM and disk space.

## Project Structure

**Dataset Files:**
ratings.dat: User-movie rating data.
users.dat: User demographic data.
movies.dat: Movie metadata.


**Implementation Files:**
- fedcollab_centralize_svd.ipynb: Performs exploratory data analysis, preprocesses the MovieLens dataset, and implements centralized SVD-based collaborative filtering using the Surprise library.
- fedcollab_cml_fl.ipynb: Implements centralized MLP training and federated learning with the RecommenderMLP model using PyTorch and Flower.
- fedcollab_fl_all_features.ipynb: Extends FL to include all dataset features (e.g., genres, demographics) for improved performance.
- fedcollab_fl_dp.ipynb: Implements FL with differential privacy, analyzing privacy-utility trade-offs.
- fedcollab_fl_latest_flower_implementation.ipynb: Uses the latest Flower framework code for FL simulations.
- fedcollab_fl_development.ipynb: Experimental file for testing and calculating precision/recall in FL.


**Generated Files:**
- preprocessed_dataset.csv: Preprocessed dataset combining ratings, users, and movies with multi-hot encoded genres and extracted years.
- clients_<N>/: Directory containing per-client CSV files (e.g., user_1.csv) for FL simulations.

## How to Run
**Step 1: Preprocess the Dataset**

- Open fedcollab_centralize_svd.py in Google Colab.
- Follow the instruction to install depenaciy and setup environment
- Upload ratings.dat, users.dat, and movies.dat when prompted.
- Run the notebook to:
  - Perform exploratory data analysis.
  - Merge datasets, extract movie years, and encode genres.
  - Save preprocessed_dataset.csv. (**This file is required for both CML and FL experiments.**)
- Download preprocessed_dataset.csv for use in other notebooks.

**Step 2: Run Centralized SVD**

- Open fedcollab_centralize_svd.py in Google Colab.
- Follow the instruction to install depenaciy and setup environment
- Ensure ratings.dat, users.dat, and movies.dat are uploaded.
- Run the notebook to:
- Train an SVD model using the Surprise library.
- Evaluate performance with MSE, RMSE, MAE, Precision@10, and Recall@10.
  - Results are printed and visualized.

**Step 3: Run Centralized MLP and Federated Learning**

- Open fedcollab_cml_fl.py in Google Colab.
- Follow the instruction to install depenaciy and setup environment
- Upload preprocessed_dataset.csv when prompted.
- Run the notebook to:
- Train a centralized MLP model on 100 users.
- Simulate FL with 100 clients over 10 rounds using Flower’s FedAvg strategy.
- Evaluate FL performance.
- Visualize MSE and RMSE trends compared to centralized baselines.
  - Results and comparison plots are displayed.

**Step 4: Run FL with All Features**

- Open fedcollab_fl_all_features.py in Google Colab.
- Follow the instruction to install depenaciy and setup environment
- Upload preprocessed_dataset.csv.
- Run the notebook to:
- Train an FL model incorporating all dataset features (e.g., genres, demographics).
- Results are printed.

**Step 5: Run FL with Differential Privacy**

- Open fedcollab_fl_dp.py in Google Colab.
- Upload preprocessed_dataset.csv.
- Run the notebook to:
  - Train an FL model with DP (ε=1.0, δ=1e-5, clip norm=1.0).
  - Evaluate privacy-utility trade-offs.
- Results and privacy impact are visualized.

**Step 6: Explore Additional FL Implementations**
  **Implemented Latest code of flower**
  - Latest Flower Implementation (fedcollab_fl_latest_flower_implementation.py):
  - Run to test FL with the latest Flower framework code updates.

  **Precesion and Recall in FL setup**
  - Development File (fedcollab_fl_development.py):
  - Run to experiment with precision/recall calculations for FL.



## Troubleshooting

- **Version Conflicts:** Ensure exact library versions (e.g., numpy==1.24.3). Restart the Colab runtime after installing dependencies.
- **File Upload Errors:** Verify that ratings.dat, users.dat, movies.dat, and preprocessed_dataset.csv are named correctly and uploaded to Colab.
- **FL Simulation Crash/Slowdown/Memory Issues:** Decrease num_rounds (e.g., to 5) or num_clients for faster execution during testing. It works max for 500 clients.
- **DP Performance Drop:** Adjust the privacy budget (ε) in fedcollab_fl_dp.py to balance privacy and accuracy (e.g., try ε=5.0 for better utility).

## Acknowledgments

- Dataset: MovieLens 1M dataset provided by GroupLens.
- Libraries: PyTorch, Flower, Surprise, NumPy, pandas, scikit-learn, matplotlib.


## Contact

For questions or issues, contact:

- Chaw Maung: [chawthm1@umbc.edu]
- Nisarg Patel: [nisargp2@umbc.edu]

## Credits

This project was designed and implemented by **Chaw Maung** and **Nisarg Patel** as part of an end-to-end Federated Learning journey in the IS-698 Federated Learning course, under the guidance of **Dr. Sanjay Purushotham**.

---

## License

This project is for educational and demonstration purposes.
