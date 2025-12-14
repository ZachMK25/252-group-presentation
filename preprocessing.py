import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw
import os

class CryptoOutlierDetector:
    """
    Detects outlier periods in cryptocurrency price data using
    time-series clustering with Dynamic Time Warping.
    """
    
    def __init__(self, window_length=14, slide_length=7, 
                 n_clusters=12, percentile=85, method='hierarchical'):
        self.window_length = window_length
        self.slide_length = slide_length
        self.n_clusters = n_clusters
        self.percentile = percentile
        self.method = method
        self.outlier_indices = []
        
    def create_subseries(self, price_series):
        """Generate overlapping subseries from price data."""
        subseries = []
        subseries_indices = []
        
        for i in range(0, len(price_series) - self.window_length + 1, 
                      self.slide_length):
            window = price_series[i:i + self.window_length]
            subseries.append(window)
            subseries_indices.append((i, i + self.window_length))
            
        return np.array(subseries), subseries_indices
    
    def normalize_subseries(self, subseries):
        """Normalize subseries using z-score normalization."""
        normalized = []
        for series in subseries:
            mean = np.mean(series)
            std = np.std(series)
            if std > 0:
                normalized.append((series - mean) / std)
            else:
                normalized.append(series - mean)
        return np.array(normalized)
    
    def compute_dtw_distances(self, subseries):
        """Compute pairwise DTW distances between all subseries."""
        n = len(subseries)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = dtw.distance(subseries[i], subseries[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        return distance_matrix
    
    def cluster_hierarchical(self, distance_matrix):
        """Perform hierarchical clustering using DTW distances."""
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')
        cluster_labels = fcluster(linkage_matrix, self.n_clusters, 
                                  criterion='maxclust')
        return cluster_labels - 1
    
    def detect_outliers_in_clusters(self, subseries, cluster_labels, 
                                   distance_matrix):
        """Detect outliers within each cluster."""
        outlier_mask = np.zeros(len(subseries), dtype=bool)
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) <= 1:
                continue
            
            # Find centroid
            cluster_distances = distance_matrix[cluster_indices][:, cluster_indices]
            avg_distances = cluster_distances.mean(axis=1)
            centroid_idx = cluster_indices[np.argmin(avg_distances)]
            
            # Compute distances from each point to centroid
            distances_to_centroid = distance_matrix[cluster_indices, centroid_idx]
            
            # Mark outliers based on percentile threshold
            threshold = np.percentile(distances_to_centroid, self.percentile)
            local_outliers = distances_to_centroid > threshold
            outlier_mask[cluster_indices[local_outliers]] = True
        
        return outlier_mask
    
    def fit(self, price_series):
        """
        Fit the outlier detector and identify outlier periods.
        """
        subseries, subseries_indices = self.create_subseries(price_series)
        
        if len(subseries) < self.n_clusters:
            print(f"    WARNING: Not enough subseries ({len(subseries)}) for {self.n_clusters} clusters")
            return np.zeros(len(price_series), dtype=bool)
        
        normalized_subseries = self.normalize_subseries(subseries)
        distance_matrix = self.compute_dtw_distances(normalized_subseries)
        
        if self.method == 'hierarchical':
            cluster_labels = self.cluster_hierarchical(distance_matrix)
        
        outlier_mask_subseries = self.detect_outliers_in_clusters(
            normalized_subseries, cluster_labels, distance_matrix
        )
        
        # Map outlier subseries back to original time series
        outlier_mask_series = np.zeros(len(price_series), dtype=bool)
        for i, is_outlier in enumerate(outlier_mask_subseries):
            if is_outlier:
                start, end = subseries_indices[i]
                outlier_mask_series[start:end] = True
        
        self.outlier_indices = np.where(outlier_mask_series)[0]
        self.subseries_indices = subseries_indices
        self.outlier_mask_subseries = outlier_mask_subseries
        self.cluster_labels = cluster_labels
        self.subseries = subseries
        
        return outlier_mask_series


def prepare_crypto_dataframe(df_raw):
    df = df_raw.copy()
    
    # Detect datetime column
    if 'datetime' in df.columns:
        dt_col = 'datetime'
    elif 'OpenDt' in df.columns:
        dt_col = 'OpenDt'
    else:
        raise ValueError('DataFrame must contain either "datetime" or "OpenDt" column')
    
    # Convert to datetime but keep the column
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col)
    
    return df


def extract_symbols_from_columns(df, dt_col='OpenDt'):
    """
    Extract unique symbol names from column names.
    """
    symbols = set()
    field_names = {'open', 'high', 'low', 'close', 'volume'}
    
    for col in df.columns:
        if col == dt_col or col == 'datetime':
            continue
        if '-' in col:
            parts = col.split('-')
            if len(parts) == 2:
                p0, p1 = parts
                if p1.lower() in field_names:
                    symbols.add(p0)
                elif p0.lower() in field_names:
                    symbols.add(p1)
    
    return sorted(list(symbols))


def detect_outliers_per_symbol(df, symbols, detector_params=None):
    """
    Apply outlier detection to each cryptocurrency symbol.
    """
    if detector_params is None:
        detector_params = {
            'window_length': 14,
            'slide_length': 7,
            'n_clusters': 9,
            'percentile': 85,
            'method': 'hierarchical'
        }
    
    dt_col = 'datetime' if 'datetime' in df.columns else 'OpenDt'
    
    results = {}
    
    print(f"\nDetecting outliers for {len(symbols)} symbols...")
    for idx, symbol in enumerate(symbols, 1):
        possible_close_cols = [
            f'{symbol}-close',
            f'close-{symbol}',
            f'{symbol}USDT-close',
            f'close-{symbol}USDT'
        ]
        
        close_col = None
        for col in possible_close_cols:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None:
            print(f"  [{idx}/{len(symbols)}] {symbol}: Close column not found, skipping")
            continue
        
        prices = df[close_col].dropna().values
        
        if len(prices) < 30:
            print(f"  [{idx}/{len(symbols)}] {symbol}: Not enough data ({len(prices)} points), skipping")
            continue
        
        print(f"  [{idx}/{len(symbols)}] {symbol}: Processing {len(prices)} data points...", end=' ')
        
        # Create detector and fit
        detector = CryptoOutlierDetector(**detector_params)
        outlier_mask = detector.fit(prices)
        
        # Get corresponding dates for outliers
        valid_indices = df[close_col].notna()
        outlier_dates = df.loc[valid_indices, dt_col].iloc[outlier_mask].values
        
        n_outliers = outlier_mask.sum()
        pct_outliers = n_outliers / len(prices) * 100
        
        print(f"Found {n_outliers} outliers ({pct_outliers:.1f}%)")
        
        # Store results
        results[symbol] = {
            'detector': detector,
            'outlier_mask': outlier_mask,
            'outlier_dates': outlier_dates,
            'n_outliers': n_outliers,
            'pct_outliers': pct_outliers,
            'close_col': close_col
        }
    
    return results


def create_clean_dataset(df, outlier_results, remove_if_any=True, save_to_csv=True, 
                        output_path='crypto_clean.csv', save_outlier_report=True):
    """
    Create a clean dataset with outlier periods removed.
    """
    # Find datetime column
    dt_col = 'datetime' if 'datetime' in df.columns else 'OpenDt'
    
    if remove_if_any:
        # Union of all outlier dates
        all_outlier_dates = set()
        for symbol, results in outlier_results.items():
            all_outlier_dates.update(results['outlier_dates'])
        
        # Remove rows where any symbol has an outlier
        df_clean = df[~df[dt_col].isin(all_outlier_dates)].copy()
        
        print(f"\n{'='*60}")
        print(f"Dataset Cleaning Summary")
        print(f"{'='*60}")
        print(f"Original rows: {len(df)}")
        print(f"Outlier rows: {len(all_outlier_dates)}")
        print(f"Clean rows: {len(df_clean)}")
        print(f"Removed: {len(all_outlier_dates)/len(df)*100:.2f}%")
        
    else:
        # Create NaN mask per symbol
        df_clean = df.copy()
        for symbol, results in outlier_results.items():
            outlier_dates = results['outlier_dates']
            
            # Get all columns for this symbol
            symbol_cols = [col for col in df.columns 
                          if symbol in col and col != dt_col]
            
            # Set outlier periods to NaN for this symbol
            mask = df[dt_col].isin(outlier_dates)
            df_clean.loc[mask, symbol_cols] = np.nan
        
        print(f"\n{'='*60}")
        print(f"Dataset Cleaning Summary (per-symbol removal)")
        print(f"{'='*60}")
        for symbol, results in outlier_results.items():
            print(f"{symbol}: {results['n_outliers']} outliers "
                  f"({results['pct_outliers']:.2f}%)")
    
    if save_to_csv:
        print(f"\nSaving clean dataset to: {output_path}")
        df_clean.to_csv(output_path, index=False)
        print(f"✓ Clean dataset saved successfully!")
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    
    if save_outlier_report:
        report_path = output_path.replace('.csv', '_outlier_report.csv')
        
        report_data = []
        for symbol, results in outlier_results.items():
            report_data.append({
                'symbol': symbol,
                'n_outliers': results['n_outliers'],
                'pct_outliers': results['pct_outliers'],
                'total_points': len(results['outlier_mask']),
                'close_column': results['close_col']
            })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df = report_df.sort_values('n_outliers', ascending=False)
            
            print(f"\nSaving outlier report to: {report_path}")
            report_df.to_csv(report_path, index=False)
            print(f"✓ Outlier report saved successfully!")
            
            print(f"\nTop 5 symbols by outlier count:")
            for idx, row in report_df.head(5).iterrows():
                print(f"  {row['symbol']}: {row['n_outliers']} outliers ({row['pct_outliers']:.1f}%)")
        
        outlier_dates_path = output_path.replace('.csv', '_outlier_dates.csv')
        outlier_dates_data = []
        
        for symbol, results in outlier_results.items():
            for date in results['outlier_dates']:
                outlier_dates_data.append({
                    'symbol': symbol,
                    'outlier_date': date
                })
        
        if outlier_dates_data:
            outlier_dates_df = pd.DataFrame(outlier_dates_data)
            outlier_dates_df = outlier_dates_df.sort_values(['symbol', 'outlier_date'])
            
            print(f"\nSaving detailed outlier dates to: {outlier_dates_path}")
            outlier_dates_df.to_csv(outlier_dates_path, index=False)
            print(f"✓ Outlier dates saved successfully!")
    
    return df_clean


def load_and_clean_crypto_data(input_path='crypto.csv', 
                               output_path='crypto_clean.csv',
                               detector_params=None,
                               save_to_csv=True,
                               save_outlier_report=True):
    """
    Convenience function to load, detect outliers, and clean crypto data in one go.
    """
    print(f"Loading data from: {input_path}")
    raw = pd.read_csv(input_path)
    
    print("Preparing dataframe...")
    df_prepared = prepare_crypto_dataframe(raw)
    
    dt_col = 'datetime' if 'datetime' in df_prepared.columns else 'OpenDt'
    
    print("Extracting symbols...")
    symbols = extract_symbols_from_columns(df_prepared, dt_col=dt_col)
    print(f"Found {len(symbols)} symbols")
    
    if detector_params is None:
        detector_params = {
            'window_length': 14,
            'slide_length': 7,
            'n_clusters': 9,
            'percentile': 85,
            'method': 'hierarchical'
        }
    
    print("\nDetector parameters:")
    for key, value in detector_params.items():
        print(f"  {key}: {value}")
    
    outlier_results = detect_outliers_per_symbol(
        df_prepared, 
        symbols,
        detector_params=detector_params
    )
    
    df_clean = create_clean_dataset(
        df_prepared, 
        outlier_results, 
        remove_if_any=True,
        save_to_csv=save_to_csv,
        output_path=output_path,
        save_outlier_report=save_outlier_report
    )
    
    return df_clean, outlier_results