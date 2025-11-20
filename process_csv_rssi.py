import pandas as pd
import numpy as np
import os
import glob

def analyze_rssi_range(csv_file_path, rssi_min=-85, rssi_max=-75):
    """
    分析CSV文件中RSSI在指定范围内的延迟和比特率数据
    
    Args:
        csv_file_path: CSV文件路径
        rssi_min: RSSI最小值（默认-85）
        rssi_max: RSSI最大值（默认-75）
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 过滤掉异常数据（延迟=9999，丢包率=100%）
        df = df[(df['latency_ms'] < 9999) & (df['loss_percent'] < 100)]
        
        # 筛选RSSI在指定范围内的数据
        rssi_filtered = df[(df['rssi_dbm'] >= rssi_min) & (df['rssi_dbm'] <= rssi_max)]
        
        if len(rssi_filtered) == 0:
            print(f"在文件 {os.path.basename(csv_file_path)} 中没有找到RSSI在{rssi_min}到{rssi_max}之间的数据")
            return None
        
        # 计算延迟和比特率的统计信息
        latency_stats = {
            'mean': rssi_filtered['latency_ms'].mean(),
            'max': rssi_filtered['latency_ms'].max(),
            'min': rssi_filtered['latency_ms'].min(),
            'std': rssi_filtered['latency_ms'].std(),
            'count': len(rssi_filtered)
        }
        
        bitrate_stats = {
            'mean': rssi_filtered['bitrate_mbps'].mean(),
            'max': rssi_filtered['bitrate_mbps'].max(),
            'min': rssi_filtered['bitrate_mbps'].min(),
            'std': rssi_filtered['bitrate_mbps'].std(),
            'count': len(rssi_filtered)
        }
        
        # 打印结果
        print(f"\n=== 文件: {os.path.basename(csv_file_path)} ===")
        print(f"RSSI范围: {rssi_min}到{rssi_max} dBm")
        print(f"数据点数: {len(rssi_filtered)}")
        
        print(f"\n延迟统计 (ms):")
        print(f"  平均值: {latency_stats['mean']:.2f}")
        print(f"  最大值: {latency_stats['max']:.2f}")
        print(f"  最小值: {latency_stats['min']:.2f}")
        print(f"  标准差: {latency_stats['std']:.2f}")
        
        print(f"\n比特率统计 (Mbps):")
        print(f"  平均值: {bitrate_stats['mean']:.2f}")
        print(f"  最大值: {bitrate_stats['max']:.2f}")
        print(f"  最小值: {bitrate_stats['min']:.2f}")
        print(f"  标准差: {bitrate_stats['std']:.2f}")
        
        return {
            'latency': latency_stats,
            'bitrate': bitrate_stats,
            'rssi_range': (rssi_min, rssi_max),
            'file': os.path.basename(csv_file_path)
        }
        
    except Exception as e:
        print(f"处理文件 {csv_file_path} 时出错: {e}")
        return None

def analyze_all_files(directory_path, rssi_min=-85, rssi_max=-75,save_path=None):
    """
    分析目录下所有CSV文件
    
    Args:
        directory_path: 目录路径
        rssi_min: RSSI最小值
        rssi_max: RSSI最大值
    """
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"在目录 {directory_path} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    all_results = []
    
    for csv_file in csv_files:
        result = analyze_rssi_range(csv_file, rssi_min, rssi_max)
        if result:
            all_results.append(result)
    
    # 汇总所有文件的结果
    if all_results:
        print("\n" + "="*50)
        print("汇总统计结果:")
        print("="*50)
        
        # 计算所有文件的平均值
        total_latency_mean = np.mean([r['latency']['mean'] for r in all_results])
        total_latency_max = np.max([r['latency']['max'] for r in all_results])
        total_bitrate_mean = np.mean([r['bitrate']['mean'] for r in all_results])
        total_bitrate_max = np.max([r['bitrate']['max'] for r in all_results])
        total_data_points = sum([r['latency']['count'] for r in all_results])
        
        print(f"总数据点数: {total_data_points}")
        print(f"涉及文件数: {len(all_results)}")
        print(f"RSSI范围: {rssi_min}到{rssi_max} dBm")
        
        print(f"\n延迟汇总 (ms):")
        print(f"  平均延迟: {total_latency_mean:.2f}")
        print(f"  最大延迟: {total_latency_max:.2f}")
        
        print(f"\n比特率汇总 (Mbps):")
        print(f"  平均比特率: {total_bitrate_mean:.2f}")
        print(f"  最大比特率: {total_bitrate_max:.2f}")
        
        # 保存详细结果到文件
        save_detailed_results(all_results, directory_path, rssi_min, rssi_max,save_path=save_path)

def save_detailed_results(results, directory_path, rssi_min, rssi_max,save_path=None):
    """
    保存详细分析结果到文件
    """
    if save_path:
        output_file = os.path.join(save_path, f"rssi_analysis_{rssi_min}to{rssi_max}.txt")
    else:
        output_file = os.path.join(directory_path, f"rssi_analysis_{rssi_min}to{rssi_max}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"RSSI范围分析报告: {rssi_min}到{rssi_max} dBm\n")
        f.write("="*50 + "\n\n")
        
        for result in results:
            f.write(f"文件: {result['file']}\n")
            f.write(f"数据点数: {result['latency']['count']}\n")
            
            f.write("延迟统计 (ms):\n")
            f.write(f"  平均值: {result['latency']['mean']:.2f}\n")
            f.write(f"  最大值: {result['latency']['max']:.2f}\n")
            f.write(f"  最小值: {result['latency']['min']:.2f}\n")
            f.write(f"  标准差: {result['latency']['std']:.2f}\n")
            
            f.write("比特率统计 (Mbps):\n")
            f.write(f"  平均值: {result['bitrate']['mean']:.2f}\n")
            f.write(f"  最大值: {result['bitrate']['max']:.2f}\n")
            f.write(f"  最小值: {result['bitrate']['min']:.2f}\n")
            f.write(f"  标准差: {result['bitrate']['std']:.2f}\n")
            f.write("-"*30 + "\n")
        
        # 汇总信息
        total_latency_mean = np.mean([r['latency']['mean'] for r in results])
        total_latency_max = np.max([r['latency']['max'] for r in results])
        total_bitrate_mean = np.mean([r['bitrate']['mean'] for r in results])
        total_bitrate_max = np.max([r['bitrate']['max'] for r in results])
        total_data_points = sum([r['latency']['count'] for r in results])
        
        f.write("\n汇总统计:\n")
        f.write(f"总数据点数: {total_data_points}\n")
        f.write(f"涉及文件数: {len(results)}\n")
        f.write(f"平均延迟: {total_latency_mean:.2f} ms\n")
        f.write(f"最大延迟: {total_latency_max:.2f} ms\n")
        f.write(f"平均比特率: {total_bitrate_mean:.2f} Mbps\n")
        f.write(f"最大比特率: {total_bitrate_max:.2f} Mbps\n")
    
    print(f"\n详细分析结果已保存到: {output_file}")

if __name__ == "__main__":
    # 设置要分析的目录
    data_directory = "//home/chenyan/demo/test/"
    save_path = "/home/chenyan/demo/useful_test/data_process"
    
    # 设置RSSI范围
    rssi_min = -85
    rssi_max = -75
    
    print(f"开始分析RSSI在{rssi_min}到{rssi_max}之间的数据...")
    analyze_all_files(data_directory, rssi_min, rssi_max,save_path=save_path)
