#!/usr/bin/env python3
"""Visualize KL divergence statistics from CSV file"""

import csv
import click
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple


def load_kld_data(csv_file: str) -> List[Tuple[int, float]]:
    """Load KLD data from CSV file"""
    data = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                context_size = int(row['context_size'])
                kl_divergence = float(row['kl_divergence'])
                data.append((context_size, kl_divergence))
    except FileNotFoundError:
        click.echo(f"Error: File '{csv_file}' not found", err=True)
        return []
    except (KeyError, ValueError) as e:
        click.echo(f"Error parsing CSV file: {e}", err=True)
        return []
    
    return data


def aggregate_by_buckets(data: List[Tuple[int, float]], bucket_size: int) -> Dict[int, List[float]]:
    """Aggregate KLD values by context position buckets"""
    buckets = defaultdict(list)
    
    for context_pos, kld in data:
        bucket = (context_pos // bucket_size) * bucket_size
        buckets[bucket].append(kld)
    
    return dict(buckets)


def calculate_statistics(buckets: Dict[int, List[float]]) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Calculate statistics for each bucket"""
    sorted_buckets = sorted(buckets.keys())
    
    bucket_centers = []
    means = []
    stds = []
    counts = []
    
    for bucket in sorted_buckets:
        values = buckets[bucket]
        bucket_centers.append(bucket)
        means.append(np.mean(values))
        stds.append(np.std(values))
        counts.append(len(values))
    
    return bucket_centers, means, stds, counts


@click.command()
@click.option('--input', '-i', default='kl_divergence_stats.csv', help='Input CSV file with KLD statistics')
@click.option('--bucket-size', '-b', default=10, type=int, help='Size of context position buckets')
@click.option('--output', '-o', help='Output file for the plot (optional)')
@click.option('--show-std', is_flag=True, help='Show standard deviation as error bars')
@click.option('--log-scale', is_flag=True, help='Use log scale for y-axis')
def main(input: str, bucket_size: int, output: str, show_std: bool, log_scale: bool):
    """Visualize KL divergence statistics by context position buckets"""
    
    # Load data
    click.echo(f"Loading data from {input}...")
    data = load_kld_data(input)
    
    if not data:
        return
    
    click.echo(f"Loaded {len(data)} data points")
    
    # Aggregate by buckets
    buckets = aggregate_by_buckets(data, bucket_size)
    bucket_centers, means, stds, counts = calculate_statistics(buckets)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot average KLD
    if show_std:
        ax1.errorbar(bucket_centers, means, yerr=stds, fmt='o-', capsize=5, 
                    label='Average KLD', color='blue', markersize=6)
    else:
        ax1.plot(bucket_centers, means, 'o-', label='Average KLD', color='blue', markersize=6)
    
    ax1.set_xlabel(f'Context Position (buckets of {bucket_size})')
    ax1.set_ylabel('Average KL Divergence', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    if log_scale:
        ax1.set_yscale('log')
    
    # Add count information on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(bucket_centers, counts, width=bucket_size*0.8, alpha=0.3, 
            label='Sample Count', color='gray')
    ax2.set_ylabel('Sample Count', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Add title and legend
    plt.title(f'KL Divergence by Context Position\n(Bucket Size: {bucket_size})')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add statistics text
    total_samples = sum(counts)
    avg_kld = np.mean([kld for _, kld in data])
    stats_text = f'Total Samples: {total_samples}\nOverall Avg KLD: {avg_kld:.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show plot
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        click.echo(f"Plot saved to {output}")
    else:
        plt.show()
    
    # Print summary statistics
    click.echo("\nSummary Statistics by Bucket:")
    click.echo(f"{'Bucket':>10} {'Count':>8} {'Avg KLD':>12} {'Std Dev':>12}")
    click.echo("-" * 45)
    for i, bucket in enumerate(bucket_centers):
        click.echo(f"{bucket:>10} {counts[i]:>8} {means[i]:>12.6f} {stds[i]:>12.6f}")


if __name__ == '__main__':
    main()