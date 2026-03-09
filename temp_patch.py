# 在 get_link_quality_by_mac 之前插入新方法

new_method = '''    def update_link_quality_cache(self, target_mac, latency=None, loss=None, rssi=None, bitrate=None):
        """
        更新特定target_mac的链路质量缓存和历史记录
        用于在获取链路信息后立即保存
        
        Args:
            target_mac (str): 目标节点的MAC地址
            latency (float): 延迟 (ms)，None表示不更新
            loss (float): 丢包率 (%)
            rssi (float): 信号强度 (dBm)
            bitrate (float): 速率 (Mbps)
        """
        # 1. 构建历史记录条目
        history_entry = {
            'mac': target_mac,
            'time': time.time(),
            'latency': latency if latency is not None else 9999.0,
            'loss': loss if loss is not None else 100.0,
            'rssi': rssi if rssi is not None else -100.0,
            'bitrate': bitrate if bitrate is not None else 0.0
        }
        
        # 2. 追加到历史记录
        self.link_quality_history.append(history_entry)
        
        # 限制历史记录大小（只保留最新1000条）
        if len(self.link_quality_history) > 1000:
            self.link_quality_history = self.link_quality_history[-1000:]
        
        # 3. 更新缓存（用于快速查询）
        self.link_quality_cache[target_mac] = history_entry.copy()
    
'''

# 在 get_latency 的返回前加入保存逻辑
save_latency = '''        
        # 为每个 IP 对应的节点保存数据到缓存
        for ip, latency_info in results.items():
            for node in nodes:
                if node.ip and node.ip.split('/')[0] == ip:
                    self.update_link_quality_cache(
                        node.mac,
                        latency=latency_info.get('latency'),
                        loss=latency_info.get('loss')
                    )
'''

print(new_method)
