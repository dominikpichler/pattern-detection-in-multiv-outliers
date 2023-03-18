import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import datetime

# TODO: Add labels for each data point

def apply_linear_feature_extraction(df_raw_data_FE: pd.DataFrame, label: str) -> dict:
	raw_data_lin = df_raw_data_FE.values
	raw_data_lin_scaled = StandardScaler().fit_transform(raw_data_lin)

	# PCA is limited to two components here. # TODO: Might be interesting to increase? maybe dynamic based on PC1 & PC2
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(raw_data_lin_scaled)
	df_principal = pd.DataFrame(data=principalComponents, columns=['PC_1', 'PC_2'])

	list_explaind_vd = pca.explained_variance_ratio_.tolist()
	expl_variance = 0
	for pc in list_explaind_vd:
		expl_variance += pc

	# Evaluate PCA results
	expl_variance_PC12 = list_explaind_vd[0] + list_explaind_vd[1]
	PCA_info_1 = ('Explained variation per principal component: <b>{} </b>'.format(pca.explained_variance_ratio_))
	PCA_info_2 = ("PC1 and PC2 cumulatively explain <b>" + str(round(expl_variance_PC12 * 100, 2)) + " %</b> of the total variance")
	PCA_info = PCA_info_1 + "<br>" + PCA_info_2

	# Create scatter plot for PCA
	fig = go.Figure(go.Scatter(x=df_principal["PC_1"], y=df_principal["PC_2"], mode="markers"))

	# Add title
	fig.update_layout(title={
		"text": "Result PCA",
		"x": 0.5  # center title
	})
	# Add axis labels
	fig.update_xaxes(title_text="\u03DB\u2081")
	fig.update_yaxes(title_text="\u03DB\u2082")

	# Set plot size
	fig.update_layout(width=600, height=800)

	# Add explanation
	fig.update_layout(annotations=[
		go.layout.Annotation(
			x=0.5,
			y=-0.3,
			showarrow=False,
			text=PCA_info,
			xref="paper",
			yref="paper",
			font=dict(size=12)
		)
	])

	# Save plot as HTML
	pcaResultHTML = "results/html/" + "PCA_for_" + label + ".html"
	pyo.plot(fig, filename=pcaResultHTML)

	label_PCA_Result_data = "results/data/" + "PCA_for_" + label + "_" + datetime.today().strftime('%Y_%m_%d') + ".csv"
	df_principal.to_csv(label_PCA_Result_data)

	dict_results = {}
	dict_results["PCA_expl_var"] = list_explaind_vd
	dict_results["PCA_Data"] = df_principal.to_dict()

	return dict_results

def apply_non_lin_feature_extraction(raw_data: pd.DataFrame, label: str,
																		 n_components: int,
																		 verbose: int,
																		 perplexity: int,
																		 n_iter: int
																		 ) -> pd.DataFrame:
	raw_data_nl = raw_data.values
	raw_data_nl_scaled = StandardScaler().fit_transform(raw_data_nl)

	tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
	tsne_results = tsne.fit_transform(raw_data_nl_scaled)

	x_1 = pd.Series(tsne_results[:, 0])
	x_2 = pd.Series(tsne_results[:, 1])
	x_3 = pd.Series(tsne_results[:, 2])

	df_tsne_result = pd.DataFrame(columns=['C1', 'C2', 'C3'])
	df_tsne_result['C1'] = x_1
	df_tsne_result['C2'] = x_2
	df_tsne_result['C3'] = x_3

	fig = go.Figure(data=[go.Scatter3d(
		x=df_tsne_result['C1'],
		y=df_tsne_result['C2'],
		z=df_tsne_result['C3'],
		mode='markers',
		marker=dict(
			size=5,
			colorscale='Viridis',  # choose a colorscale
			opacity=0.8
		)
	)])
	# Set plot size
	fig.update_layout(width=700, height=600)

	# Set title
	fig.update_layout(title={
		"text": "Result t-SNE in 3D",
		"x": 0.5  # center title
	})

	# Set axis labels
	fig.update_layout(scene=dict(xaxis_title='\u03C3\u2081', yaxis_title='\u03C3\u2082', zaxis_title='\u03C3\u2083'))

	# Save plot as HTML
	pcaResultHTML = "results/html/" + "tSNE_for_" + label + "_" + datetime.today().strftime('%Y_%m_%d') + ".html"
	pyo.plot(fig, filename=pcaResultHTML)

	label_tsne_Result_data = "results/data/" + "t-SNE_for_" + label + "_" + datetime.today().strftime('%Y_%m_%d') + ".csv"
	df_tsne_result.to_csv(label_tsne_Result_data)

	dict_results = {}
	dict_results["tSNE_Data"] = df_tsne_result.to_dict()
	return dict_results
