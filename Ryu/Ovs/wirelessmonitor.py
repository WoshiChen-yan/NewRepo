from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link

class WirelessMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(WirelessMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.port_stats = {}
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """处理交换机状态变化"""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.logger.info(f'注册数据路径: {datapath.id}')
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                self.logger.info(f'注销数据路径: {datapath.id}')

    def _monitor(self):
        """定期获取端口统计信息"""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)  # 每10秒更新一次

    def _request_stats(self, datapath):
        """请求端口统计信息"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """处理端口统计信息回复"""
        body = ev.msg.body
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        if dpid not in self.port_stats:
            self.port_stats[dpid] = {}

        for stat in body:
            port_no = stat.port_no
            self.port_stats[dpid][port_no] = {
                'rx_packets': stat.rx_packets,
                'tx_packets': stat.tx_packets,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_errors': stat.rx_errors,
                'tx_errors': stat.tx_errors,
                'rx_dropped': stat.rx_dropped,
                'tx_dropped': stat.tx_dropped
            }
            
            # 打印统计信息
            self.logger.info(f'端口统计信息 dpid={dpid} port_no={port_no}')
            self.logger.info(f'  接收: {stat.rx_packets} 包 ({stat.rx_bytes} 字节)')
            self.logger.info(f'  发送: {stat.tx_packets} 包 ({stat.tx_bytes} 字节)')
            self.logger.info(f'  丢包: rx={stat.rx_dropped} tx={stat.tx_dropped}')

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """处理交换机加入事件"""
        switch = ev.switch
        self.logger.info(f'交换机加入: {switch.dp.id}')
        # 获取端口信息
        for port in switch.ports:
            self.logger.info(f'  端口: {port.port_no} - {port.hw_addr}')