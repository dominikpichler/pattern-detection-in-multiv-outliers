import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
import plotly.graph_objs as go


# Needed for dynamic of number of clusters
plot_colors = [
	'#1f77b4',  # muted blue
	'#ff7f0e',  # safety orange
	'#2ca02c',  # cooked asparagus green
	'#d62728',  # brick red
	'#9467bd',  # muted purple
	'#8c564b',  # chestnut brown
	'#e377c2',  # raspberry yogurt pink
	'#7f7f7f',  # middle gray
	'#bcbd22',  # curry yellow-green
	'#17becf'  # blue-teal
]

def apply_kMeans_clustering_2D(raw_data: pd.DataFrame, label: str) -> dict:
	n_max = 2
	n_dim = len(raw_data.columns)
	if n_dim > n_max:
		raise ValueError("Number of Dimensions of handed Data is too high, please apply a dimension reduction first")

	np_raw_data_values = raw_data.values

	# --- Defining the optimal number of clusters K using the Silhouette Method ---
	sil = []
	kmax = 10
	k_start = 2

	for k in range(k_start, kmax + 1):
		kmeans_sil = KMeans(n_clusters=k).fit(np_raw_data_values)
		labels = kmeans_sil.labels_
		sil.append(silhouette_score(np_raw_data_values, labels, metric='euclidean'))

	k_algorithmic = k_start + sil.index(max(sil))

	kmeans = KMeans(
		n_clusters=k_algorithmic,
		init="random",
		random_state=0,
		n_init=15,
		max_iter=100,
	)

	# Array with index that indicates affiliations to cluster
	cluster_km = kmeans.fit_predict(np_raw_data_values)
	df_result_cluster = raw_data.copy()
	df_result_cluster['Cluster'] = pd.Series(cluster_km)

	# --- Plotting the clusters --- :
	fig = go.Figure()
	fig.update_layout(
		xaxis_title="Dim1",
		yaxis_title="Dim2",
		title=label,
		showlegend=True,
	)

	i = 0
	while i <= max(cluster_km):
		fig.add_trace(go.Scatter(
			x=np_raw_data_values[cluster_km == i, 0],
			y=np_raw_data_values[cluster_km == i, 1],
			mode='markers',
			marker=dict(symbol='square', size=10, color=plot_colors[i], line=dict(width=1, color='black')),
			name='cluster ' + str(i)
		))
		i += 1

	# Calculate OLS for eac Cluster to evaluate performance:
	dict_regression_summary = {}
	dict_regression_models = {}
	i = 0
	while i <= max(cluster_km):
		dict_regression_models["regression_cluster_" + str(i)] = sm.OLS(np_raw_data_values[cluster_km == i, 1], sm.add_constant(np_raw_data_values[cluster_km == i, 0])).fit()
		dict_regression_summary["regression_cluster_" + str(i)] = sm.OLS(np_raw_data_values[cluster_km == i, 1], sm.add_constant(np_raw_data_values[cluster_km == i, 0])).fit().summary()
		i += 1

	# Add Centroids of cluster to plot
	fig.add_trace(go.Scatter(
		x=kmeans.cluster_centers_[:, 0],
		y=kmeans.cluster_centers_[:, 1],
		mode='markers',
		marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='black')),
		name='centroids'
	))


	# --- Output ---
	clusteringResult = "results/html/" + "k-Means_Clustering_for_" + label + "_" + datetime.today().strftime('%Y_%m_%d') + ".html"
	fig.write_html(clusteringResult)

	dict_results = {}
	dict_results["regression"] = dict_regression_summary
	dict_results["clustering"] = df_result_cluster.to_dict()

	label_k_means_clustering = "results/data" + "k-Means_Clustering_for_" + label + "_" + datetime.today().strftime('%Y_%m_%d') + ".csv"
	df_result_cluster.to_csv(label_k_means_clustering)

	return dict_results
