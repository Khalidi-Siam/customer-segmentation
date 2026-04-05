import pandas as pd

def build_cluster_profile_and_labels(df_with_cluster: pd.DataFrame):
	profile_df = (
		df_with_cluster
		.groupby('cluster', as_index=False)
		.agg(
			customer_count=('cluster', 'size'),
			avg_age=('Age', 'mean'),
			avg_income=('Annual Income (k$)', 'mean'),
			avg_spending=('Spending Score (1-100)', 'mean'),
		)
		.sort_values('cluster')
	)

	income_q70 = profile_df['avg_income'].quantile(0.70)
	income_q30 = profile_df['avg_income'].quantile(0.30)
	spending_q70 = profile_df['avg_spending'].quantile(0.70)
	spending_q30 = profile_df['avg_spending'].quantile(0.30)

	cluster_name_map = {}
	for _, row in profile_df.iterrows():
		cluster_id = int(row['cluster'])
		income = row['avg_income']
		spending = row['avg_spending']

		if income >= income_q70 and spending >= spending_q70:
			segment_name = 'High Value'
		elif income >= income_q70 and spending <= spending_q30:
			segment_name = 'Careful Affluent'
		elif income <= income_q30 and spending >= spending_q70:
			segment_name = 'High Spend Low Income'
		elif income <= income_q30 and spending <= spending_q30:
			segment_name = 'Budget Conservative'
		else:
			segment_name = 'Mainstream'

		cluster_name_map[cluster_id] = segment_name

	profile_df['segment_name'] = profile_df['cluster'].astype(int).map(cluster_name_map)
	return profile_df, cluster_name_map
