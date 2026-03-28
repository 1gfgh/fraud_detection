from collections import defaultdict
from copy import deepcopy
from typing import Optional, List, Dict
from graph.utils import get_string_hash
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs.log'
)
logger = logging.getLogger('graph')


class Node():
    def __init__(self, node: Optional[dict] = None):
        self.id: str = 'root'
        self.data = {}
        self.size: int = 0
        self.children: List[str] = list()
        self.transactions: List[str] = list()
        if node:
            self.id = node['id']
            self.data = node['data']
            self.size = node['size']
            self.children = node['children']
            self.transactions = node['transactions']

    def dump(self) -> dict:
        node = {
            'id': self.id,
            'data': self.data,
            'size': self.size,
            'children': self.children,
            'transactions': self.transactions,
        }
        return node

    def create_id(self, parent_id: str) -> None:
        """
        create and set ID from parent's ID and data.values()
        """
        vals_str = str(self.data.items())
        new_id_str = parent_id + vals_str
        self.id = get_string_hash(new_id_str)


class Graph():
    def __init__(self, build_params: dict):
        self.id: str = None
        self.build_params = build_params
        self.nodes: Dict[str, Node] = {}

    def to_dict(self) -> dict:
        nodes = deepcopy(self.nodes)
        for node_id in nodes:
            nodes[node_id] = nodes[node_id].dump()

        result = {
            "id": self.id,
            "build_params": self.build_params,
            "nodes": nodes
        }
        return result

    @classmethod
    def from_dict(cls, data) -> 'Graph':
        id = data.get('id')
        build_params = data.get('build_params')
        nodes = data.get('nodes')

        for node_id in nodes:
            nodes[node_id] = Node(nodes[node_id])

        graph = cls()
        graph.build_params = build_params
        graph.id = id
        graph.nodes = nodes
        return graph

    def create_id(self, data_source_id: str, data_source_time: str):
        build_str = str(self.build_params.values())
        new_id_str = data_source_id + build_str + data_source_time
        self.id = get_string_hash(new_id_str)

    def __deepcopy__(self, memo):
        new_obj = Graph()
        new_obj.id = self.id
        new_obj.build_params = deepcopy(self.build_params, memo)
        new_obj.nodes = deepcopy(self.nodes, memo)
        return new_obj

    def _get_row_group_value(self, row: dict, keys: tuple) -> tuple:
        return tuple(row.get(key, None) for key in keys)

    def _group_transactions(self, transactions: List[dict], keys: tuple) -> dict:
        logger.info(f"Group up transaction by key {keys}")
        grouped_trans = defaultdict(list)
        for transaction in transactions:
            group_value = self._get_row_group_value(transaction, keys)
            hashed_value = get_string_hash(str(group_value))
            grouped_trans[hashed_value].append(transaction)
        return grouped_trans

    def _process_route(
            self,
            transactions: List[dict],
            node_keys: tuple) -> None:
        cur_node_id = "root"
        transactions.sort(key=lambda transaction: transaction['TransactionStartTime'])

        logger.info(f"Process route of {len(transactions)} transactions")
        for transaction in transactions:
            new_node = Node()
            increase_size = False
            for key in node_keys:
                new_node.data[key] = transaction[key]
            if new_node.data == self.nodes[cur_node_id].data:
                new_node = self.nodes[cur_node_id]
            else:
                new_node.create_id(cur_node_id)
                increase_size = True
            new_node_id = new_node.id

            if new_node_id not in self.nodes:
                logger.info("Node is unknown! Create new node.")
                self.nodes[new_node_id] = deepcopy(new_node)
                self.nodes[cur_node_id].children.append(new_node_id)

            if increase_size:
                self.nodes[new_node_id].size += 1
            logger.info(f"Size of node {new_node_id} is {self.nodes[new_node_id].size}")

            cur_trid = transaction['TransactionId']
            self.nodes[new_node_id].transactions.append(cur_trid)
            logger.info(f"Added transaction with Transaction ID {cur_trid} to node {new_node_id}")

            cur_node_id = new_node_id
        logger.info(f"All of transactions are processed!")

    def build(
            self,
            transactions: List[dict]) -> Dict[str, Node]:
        logger.info(f"Start building the graph!")
        transactions.sort(key=lambda transaction: transaction['TransactionStartTime'])
        logger.info(f"Found {len(transactions)} transactions")

        edge_keys = tuple(self.build_params.get('edge_keys', []))
        grouped_trans = self._group_transactions(transactions, edge_keys)

        self.nodes["root"] = self.nodes.get("root", Node())
        for group_id, trans in grouped_trans.items():
            logger.info(f"Process route of group {group_id}")
            self._process_route(trans, self.build_params['node_keys'])
            logger.info(f"Route of group {group_id} is processed!")
        return self.nodes
