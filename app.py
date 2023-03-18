import pandas as pd
import matplotlib
import json

# Custom, locally stored Packages, just for better structuring
from helper_calculations.clustering import apply_kMeans_clustering_2D
from helper_calculations.feature_extraction import apply_linear_feature_extraction, apply_non_lin_feature_extraction
from helper_calculations.fourier_transformation import apply_fourier_transformation
from helper_visualization.helper_plotting import plot_in_3D


# # General Settings for plots
matplotlib.use("macosx")
matplotlib.rcParams.update({
 	'font.family': 'serif',
 	'text.usetex': False,
 	'pgf.rcfonts': False,
 })

# TODO: Folderstructure for results
# TODO: Automatic Labeling @ Plots
# TODO: Multi-Dim-Cluster!!!
# TODO: Opt: Sampling Frequency in Fourier
# TODO: Fourier for each Cluster
# TODO: Make app scaleable

# Function that checks the provided raw data to catch potential exceptions/problems in advance.
def load_rawData(path: str) -> pd.DataFrame:
	df_rawData = pd.read_csv(path,sep=",")
	# TODO: Tests for data quality
	return df_rawData


def find_outliers_in_raw_data(df_raw_data:pd.DataFrame):
	print("Please integrate the mahalanobis distance here :) ")
	# TODO: Do me :)
# Function that runs the entire algorithm based on the program-graph as shown in the readme file
def find_patterns_in_outlier(df_outlier_data: pd.DataFrame, label) -> dict:
	dict_solutions = {}
	dict_result_clustering = {}
	dict_result_fourier = {}

	# ======== Step 0: General Data Preperation ========
	print(" ==== Step 0: General Data Preperation ====")
	# 1. Branch -  No Transformation/Initial Feature Extraction. Just Fourier- and k-Means Clustering:
	df_sol_non_trans = df_outlier_data.copy()

	# 2. Branch - Initial Linear Transformation (PCA):
	if len(df_outlier_data.columns) > 2:
		print(" --- Running Linear Transformation (PCA) ---")
		df_sol_lin_trans = apply_linear_feature_extraction(df_outlier_data, label)
	else:
		df_sol_lin_trans = df_outlier_data

	# 3. Branch - Initial Non-Linear Transformation (tSNE)
	# Checking the dimensions. Transformations only make sense when the data set has more then 2 dimensions (columns)
	if len(df_outlier_data.columns) > 2:
		print(" --- Running Non-Linear Transformation (t-SNE) ---")
		df_sol_non_lin_trans = apply_non_lin_feature_extraction(df_outlier_data, label)
	else:
		df_sol_non_lin_trans = df_outlier_data # TODO: Isn't this stupid? (Doppeltgemoppelt)

	# ======== Step 1: Clustering ========
	print(" ==== Step 1: Clustering ====")

	# Clustering on raw data if raw data has just two dimensions:
	if len(df_outlier_data.columns) == 2:
		print(" --- Running Clustering on raw data --- ")
		label_nt = label + "[NonTrans]"
		df_cluster_nonTrans = apply_kMeans_clustering_2D(df_sol_non_trans, label_nt)
		dict_result_clustering["nonTrans"] = df_cluster_nonTrans

	elif len(df_outlier_data.columns) > 2:
		# Clustering on linearly transformed raw data:
		print(" --- Running clustering on linearly transformed raw data --- ")
		label_lt = label + "[LinTrans]"
		df_cluster_linTrans = apply_kMeans_clustering_2D(df_sol_lin_trans, label_lt)
		dict_result_clustering["linTrans"] = df_cluster_linTrans

		# Clustering on non-linearly transformed raw data:
		print(" --- Running clustering on non-linearly transformed raw data --- ")
		label_nlt = label + "[NonLinTrans]"
		df_cluster_nonLinTrans = apply_kMeans_clustering_2D(df_sol_non_lin_trans, label_nlt)
		dict_result_clustering["nonLinTrans"] = df_cluster_nonLinTrans

	else:
		# TODO: Find solution for this case. Clustering should also be possible for 1D. And raw data with a count of dimensions equal to zero should be catched in advance anyway
		raise Exception("Number of Exceptions is too low. (Has to be >= 2")

	# TODO: Generalize a permuation for n dimensions. Might be usefull for manual investigations
	if len(df_outlier_data.columns == 3):
		# Variational clustering along all axis:
		df_rawData_dim1_2 = pd.DataFrame(columns=["1", "2"])
		df_rawData_dim1_2["1"] = df_outlier_data.iloc[:, 0]
		df_rawData_dim1_2["2"] = df_outlier_data.iloc[:, 1]
		label12 = label + "clustering_dim_12"
		apply_kMeans_clustering_2D(df_rawData_dim1_2, label12)
		apply_fourier_transformation(df_rawData_dim1_2, label12)

		df_rawData_dim2_3 = pd.DataFrame(columns=["2", "3"])
		df_rawData_dim2_3["2"] = df_outlier_data.iloc[:, 1]
		df_rawData_dim2_3["3"] = df_outlier_data.iloc[:, 2]
		label23 = label + "clustering_dim_23"
		apply_kMeans_clustering_2D(df_rawData_dim2_3, label23)
		apply_fourier_transformation(df_rawData_dim2_3, label23)

		df_rawData_dim1_3 = pd.DataFrame(columns=["1", "3"])
		df_rawData_dim1_3["1"] = df_outlier_data.iloc[:, 0]
		df_rawData_dim1_3["3"] = df_outlier_data.iloc[:, 2]
		label13 = label + "clustering_dim_13"
		apply_kMeans_clustering_2D(df_rawData_dim1_3, label13)
		apply_fourier_transformation(df_rawData_dim1_3, label13)

	dict_solutions["Clustering"] = dict_result_clustering

	# ======== Step 2: Fourier Analysis/Cluster ========
	print(" ==== Step 2: Fourier Analysis ====")

	if (len(df_sol_non_trans.columns) == 2):
		# Fourier Analysis on raw data in case the raw data has just two dimensions:
		print(" --- Running Fourier Analysis on raw data --- ")
		label_nt = label + "[NonTrans]"
		df_fourier_nonTrans = apply_fourier_transformation(df_sol_non_trans, label_nt)
		dict_result_fourier["nonTrans"] = df_fourier_nonTrans

	if len(df_outlier_data.columns) > 2:
		# Fourier Analysis on linearly transformed raw data:
		print(" --- Running Fourier Analysis on linearly transformed raw data --- ")
		label_lt = label + "[LinTrans]"
		df_fourier_linTrans = apply_fourier_transformation(df_sol_lin_trans, label_lt)
		dict_result_fourier["linTrans"] = df_fourier_linTrans

		# Fourier Analysis on non-linearly transformed raw data:
		print(" --- Running Fourier Analysis on non-linearly transformed raw data --- ")
		label_nlt = label + "[NonLinTrans]"
		df_fourier_nonLinTrans = apply_fourier_transformation(df_sol_non_lin_trans, label_nlt)
		dict_result_fourier["nonLinTrans"] = df_fourier_nonLinTrans

	dict_solutions["Fourier"] = dict_result_fourier

	# ======== Step 3: Storing/Handling results ========
	print(" ==== Step 3: Storing results ====")
	with open('results/result_full.json', 'w') as fp:
		json.dump(dict_solutions, fp)

	return dict_solutions
	print(" ==== DONE ====")

if __name__ == '__main__':
	df_raw_data = load_rawData("raw_data/kMeans.csv")

	two_dim = load_rawData("raw_data/2D.csv")
	three_dim = load_rawData("raw_data/3D.csv")
	four_dim = load_rawData("raw_data/4D.csv")

	df_raw_data = df_raw_data.iloc[:, 1:]

	#apply_linear_feature_extraction(four_dim,"testing_purposes")
	apply_non_lin_feature_extraction(four_dim,"testing_purposes", n_components=3,verbose=1,perplexity=40,n_iter=300)
	#apply_kMeans_clustering_2D(two_dim,"testing_purposes")
	#find_patterns_in_outlier(df_raw_data, "_outliers_")
	plot_in_3D(three_dim)
