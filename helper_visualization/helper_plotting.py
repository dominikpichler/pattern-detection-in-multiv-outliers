import pandas as pd
import matplotlib
import plotly.graph_objs as go
from datetime import datetime

# Plots a given 3D dataset in an interactive plotly plot that is then stored as .html for further processing
def plot_in_3D(df_rawData: pd.DataFrame):
	x = df_rawData.iloc[:, 0]
	y = df_rawData.iloc[:, 1]
	z = df_rawData.iloc[:, 2]
	v = df_rawData.iloc[:, 3]

	# Create the trace for the 3D scatter plot
	trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8), name='Data Points', text=[v[i] for i in range(len(v))])

	# Create the layout for the plot
	layout = go.Layout(scene=dict(xaxis=dict(title='\u03BE\u2081'), yaxis=dict(title='\u03BE\u2082'), zaxis=dict(title='\u03BE\u2083')), legend=dict(title='Clusters'))
	# Create the figure object and add the trace and layout
	fig = go.Figure(data=[trace], layout=layout)

	# Set plot size
	fig.update_layout(width=800, height=650)

	# Set title
	fig.update_layout(title={
		"text": "3D Plot",
		"x": 0.5  # center title
	})
	# Save the plot to an HTML file
	fig.write_html("results/3D_plot_" + datetime.today().strftime('%Y_%m_%d') + ".html")
