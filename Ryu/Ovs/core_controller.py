from ryu.base import app_manager
from ryu.controller import ofp_event    
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_switch, get_link

class Corecontroller(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Corecontroller, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.topology_api_app = self
        self.net = {}

    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):
        switches= get_switch(self.topology_api_app, None)
        links=get_link(self.topology_api_app, None)
        self.net= {sw.dp.id: set() for sw in switches}
        for link in links:
            self.net[link.src.dpid].add(link.dst.dpid)
        self.logger.info(f"Net Topology = {self.net}")   