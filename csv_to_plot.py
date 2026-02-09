#!/usr/bin/env python3
"""
从CSV文件绘制链路质量图的工具
功能类似于net.py中的plot_node_link_quality方法
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
from datetime import datetime

class CSVLinkQualityPlotter:
    def __init__(self):
        self.data = {}
    
    def load_csv_file(self, csv_file_path):
        """加载单个CSV文件"""
        try:
            df = pd.read_csv(csv_file_path)
            print(f"成功加载文件: {csv_file_path}")
            print(f"数据行数: {len(df)}")
            print(f"数据列: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"加载文件 {csv_file_path} 失败: {e}")
            return None
    
    def load_all_csv_files(self, directory_path):
        """加载目录下所有CSV文件"""
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        all_data = {}
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            df = self.load_csv_file(csv_file)
            if df is not None:
                all_data[filename] = df
                # 提取节点信息
                if '_to_' in filename:
                    parts = filename.split('_to_')
                    src_node = parts[0]
                    target_part = parts[1].split('_data_')[0]
                    key = f"{src_node}_to_{target_part}"
                    self.data[key] = df
        
        print(f"总共加载了 {len(all_data)} 个CSV文件")
        return all_data
    
    def plot_link_quality_from_csv(self, csv_file_path, save_path=None, 
                                  rssi_range=None, bitrate_range=None, 
                                  loss_range=None, latency_range=None,
                                  filter_anomalies=True):
        """
        从CSV文件绘制链路质量图
        参数与net.py中的plot_node_link_quality方法类似
        """
        df = self.load_csv_file(csv_file_path)
        if df is None:
            return
        
        # 提取节点信息
        filename = os.path.basename(csv_file_path)
        if '_to_' in filename:
            parts = filename.split('_to_')
            src_node = parts[0]
            target_part = parts[1].split('_data_')[0]
            src_node_name = src_node
            target_node_name = target_part
        else:
            src_node_name = "Unknown"
            target_node_name = "Unknown"
        
        # 过滤异常数据（可选）
        if filter_anomalies:
            # 过滤掉延迟为9999ms或丢包率为100%的数据
            df_clean = df[(df['latency_ms'] != 9999.0) & (df['loss_percent'] != 100.0)]
            if len(df_clean) == 0:
                print("过滤后无有效数据，使用原始数据")
                df_clean = df
        else:
            df_clean = df
        
        if len(df_clean) == 0:
            print(f"文件 {csv_file_path} 中无有效数据")
            return
        
        # 提取数据
        times = df_clean['measurement_time'].values
        rssis = df_clean['rssi_dbm'].values
        bitrates = df_clean['bitrate_mbps'].values
        loss = df_clean['loss_percent'].values
        latency = df_clean['latency_ms'].values
        
        print(f"绘制节点 {src_node_name} 到 {target_node_name} 的链路质量图")
        print(f"有效数据点数: {len(times)}")
        print(f"RSSI范围: {rssis.min():.1f} 到 {rssis.max():.1f} dBm")
        print(f"比特率范围: {bitrates.min():.1f} 到 {bitrates.max():.1f} Mbps")
        print(f"延迟范围: {latency.min():.2f} 到 {latency.max():.2f} ms")
        print(f"丢包率范围: {loss.min():.1f} 到 {loss.max():.1f} %")
        
        # 创建图表
        plt.figure(figsize=(16, 14))
        
        # 1. RSSI图
        plt.subplot(2, 1, 1)
        plt.plot(times, rssis, marker='o')
        plt.title(f"Node {src_node_name} to Node {target_node_name} RSSI over time")
        plt.ylabel("RSSI (dBm)")
        plt.xlabel("Measurement Time")
        # plt.grid(True, alpha=0.3)
        if rssi_range:
            plt.ylim(rssi_range)
        
        # # 2. 比特率图
        # plt.subplot(4, 1, 2)
        # plt.plot(times, bitrates, marker='o', color='orange')
        # plt.title(f"Node {src_node_name} to Node {target_node_name} Bitrate over time")
        # plt.ylabel("Bitrate (Mbps)")
        # plt.xlabel("Measurement Time")
        # # plt.grid(True, alpha=0.3)
        # if bitrate_range:
        #     plt.ylim(bitrate_range)
        
        # # 3. 丢包率图
        # plt.subplot(4, 1, 3)
        # plt.plot(times, loss, marker='o', color='red')
        # plt.title(f"Node {src_node_name} to Node {target_node_name} Packet Loss over time")
        # plt.ylabel("Loss (%)")
        # plt.xlabel("Measurement Time")
        # # plt.grid(True, alpha=0.3)
        # if loss_range:
        #     plt.ylim(loss_range)
        
        # 4. 延迟图
        plt.subplot(2, 1, 2)
        plt.plot(times, latency, marker='o', color='green')
        plt.title(f"Node {src_node_name} to Node {target_node_name} Latency over time")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Measurement Time")
        # plt.grid(True, alpha=0.3)
        if latency_range:
            plt.ylim(latency_range)
        
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            output_filename = f"{src_node_name}_to_{target_node_name}_link_quality.png"
            output_path = os.path.join(save_path, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
        else:
            output_dir = "csv_plots"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{src_node_name}_to_{target_node_name}_link_quality.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
        
        plt.close()
        
        # 保存统计数据
        self._save_statistics(df_clean, src_node_name, target_node_name, save_path)
    
    def plot_all_links_from_directory(self, directory_path, save_path=None, filter_anomalies=True):
        """绘制目录下所有CSV文件的链路质量图"""
        self.load_all_csv_files(directory_path)
        
        for key, df in self.data.items():
            if '_to_' in key:
                parts = key.split('_to_')
                src_node = parts[0]
                target_node = parts[1]
                
                # 创建临时文件路径用于绘图
                temp_csv_path = os.path.join(directory_path, f"{src_node}_to_{target_node}_data_*.csv")
                csv_files = glob.glob(temp_csv_path)
                if csv_files:
                    self.plot_link_quality_from_csv(csv_files[0], save_path, filter_anomalies=filter_anomalies)
    
    def _save_statistics(self, df, src_node, target_node, save_path=None):
        """保存统计数据到文本文件"""
        stats = {
            '数据点数': len(df),
            'RSSI最小值': df['rssi_dbm'].min(),
            'RSSI最大值': df['rssi_dbm'].max(),
            'RSSI平均值': df['rssi_dbm'].mean(),
            '比特率最小值': df['bitrate_mbps'].min(),
            '比特率最大值': df['bitrate_mbps'].max(),
            '比特率平均值': df['bitrate_mbps'].mean(),
            '延迟最小值': df['latency_ms'].min(),
            '延迟最大值': df['latency_ms'].max(),
            '延迟平均值': df['latency_ms'].mean(),
            '丢包率最小值': df['loss_percent'].min(),
            '丢包率最大值': df['loss_percent'].max(),
            '丢包率平均值': df['loss_percent'].mean()
        }
        
        if save_path:
            stats_file = os.path.join(save_path, f"{src_node}_to_{target_node}_statistics.txt")
        else:
            stats_dir = "csv_plots"
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir, exist_ok=True)
            stats_file = os.path.join(stats_dir, f"{src_node}_to_{target_node}_statistics.txt")
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"节点 {src_node} 到 {target_node} 链路质量统计数据\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"统计数据已保存到: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='从CSV文件绘制链路质量图')
    parser.add_argument('--file', type=str, help='单个CSV文件路径')
    parser.add_argument('--directory', type=str, help='包含CSV文件的目录路径')
    parser.add_argument('--save-path', type=str, help='保存图表的目录路径')
    parser.add_argument('--no-filter', action='store_true', help='不过滤异常数据')
    
    args = parser.parse_args()
    
    plotter = CSVLinkQualityPlotter()
    
    if args.file:
        # 处理单个文件
        plotter.plot_link_quality_from_csv(
            args.file, 
            args.save_path,
            filter_anomalies=not args.no_filter
        )
    elif args.directory:
        # 处理目录下所有文件
        plotter.plot_all_links_from_directory(
            args.directory,
            args.save_path,
            filter_anomalies=not args.no_filter
        )
    else:
        # 如果没有指定文件或目录，使用默认的test目录
        default_dir = "test"
        if os.path.exists(default_dir):
            print(f"使用默认目录: {default_dir}")
            plotter.plot_all_links_from_directory(
                default_dir,
                args.save_path,
                filter_anomalies=not args.no_filter
            )
        else:
            print("请指定CSV文件路径或目录路径")
            print("用法示例:")
            print("  python plot_csv_link_quality.py --file test/11_to_12_data_20251120_212035.csv")
            print("  python plot_csv_link_quality.py --directory test")
            print("  python plot_csv_link_quality.py --directory test --save-path my_plots")

if __name__ == "__main__":
    main()