import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
import textwrap
from typing import Dict, Optional, Tuple

class SeparatedKnowledgeGraphs:
    def __init__(self):
        self.base_dir = Path("Knowledge_Graphs")
        self.story_graphs_dir = self.base_dir / "Life_Story_Graphs"
        self.routine_graphs_dir = self.base_dir / "Daily_Routine_Graphs"
        
        for directory in [self.story_graphs_dir, self.routine_graphs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.node_colors = {
            'person': '#66c2a5',    # 绿色
            'event': '#fc8d62',     # 橙色
            'activity': '#8da0cb'   # 蓝色
        }
    
    def create_story_graph(self, file_path: Path) -> Optional[Tuple[nx.Graph, str]]:
        """创建生平故事知识图谱"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            G = nx.Graph()
            patient_id = data['profile']['patient_id']
            
            # 添加病人节点
            G.add_node(patient_id,
                    node_type='person',
                    role='patient',
                    age=data['profile']['age'],
                    career=data['profile']['career'],
                    education=data['profile']['education_level'],
                    display_name=f"Patient\n{patient_id}")
            
            # 处理访谈数据
            for idx, interview in enumerate(data['interviews']):
                interviewer = interview['interviewee']
                interviewer_id = f"INT_{idx}"
                
                # 添加访谈者节点
                G.add_node(interviewer_id,
                        node_type='person',
                        role='interviewer',
                        name=interviewer['name'],
                        relationship=interviewer['relationship'],
                        display_name=f"{interviewer['name']}\n({interviewer['relationship']})")
                
                # 连接访谈者和病人
                G.add_edge(patient_id, interviewer_id,
                        relationship=interviewer['relationship'])
                
                # 添加记忆事件节点
                for memory_idx, memory in enumerate(interview['memories']):
                    event_id = f"MEM_{idx}_{memory_idx}"
                    
                    # 使用 title 和年份创建显示名称
                    display_name = f"{memory['title']}\n({memory['year']})"
                    
                    G.add_node(event_id,
                            node_type='event',
                            year=memory['year'],
                            title=memory['title'],
                            description=memory['description'],
                            impact=memory['impact'],
                            details=memory.get('details', ''),
                            display_name=display_name)
                    
                    G.add_edge(patient_id, event_id, relation='experienced')
                    G.add_edge(interviewer_id, event_id, relation='recalled')
            
            return G, patient_id
            
        except Exception as e:
            print(f"Error processing story file {file_path}: {e}")
            return None 
    

    def create_routine_graph(self, file_path: Path) -> Optional[Tuple[nx.Graph, str]]:
        """创建日常活动知识图谱"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            G = nx.Graph()
            activities = data.get('activities', [])
            if not activities:
                return None
            
            patient_id = Path(file_path).stem.split('_')[0]
            
            # 添加病人节点
            G.add_node(patient_id,
                      node_type='person',
                      role='patient',
                      display_name=f"Patient\n{patient_id}")
            
            # 跟踪已添加的人物
            added_people = set()
            
            # 添加活动节点
            for idx, activity in enumerate(activities):
                activity_id = f"ACT_{idx}"
                display_name = f"{activity['time_start']}-{activity['time_end']}\n{activity['activity']}"
                
                G.add_node(activity_id,
                          node_type='activity',
                          time=f"{activity['time_start']}-{activity['time_end']}",
                          activity_type=activity['activity'],
                          location=activity['location'],
                          details=activity['details'],
                          display_name=display_name)
                
                # 连接活动到病人
                G.add_edge(patient_id, activity_id, relation='participates_in')
                
                # 添加参与者节点
                for participant in activity['participants']:
                    person_id = f"PART_{participant['role']}_{participant['name']}"
                    
                    if person_id not in added_people:
                        G.add_node(person_id,
                                 node_type='person',
                                 role=participant['role'],
                                 name=participant['name'],
                                 display_name=f"{participant['name']}\n({participant['role']})")
                        added_people.add(person_id)
                    
                    G.add_edge(person_id, activity_id, relation='involved_in')
            
            return G, patient_id
            
        except Exception as e:
            print(f"Error processing routine file {file_path}: {e}")
            return None

    def visualize_graph(self, G: nx.Graph, output_file: Path, title: str):
        """生成知识图谱可视化"""
        plt.figure(figsize=(20, 15))
        
        # 使用spring_layout并增加节点间距
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 绘制不同类型的节点
        for node_type, color in self.node_colors.items():
            nodes = [node for node, attr in G.nodes(data=True)
                    if attr['node_type'] == node_type]
            if nodes:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=nodes,
                                     node_color=color,
                                     node_size=2000 if node_type == 'person' else 1500,
                                     alpha=0.6,
                                     label=node_type)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
        
        # 添加标签
        labels = {node: G.nodes[node]['display_name'] 
                 for node in G.nodes()}
        
        nx.draw_networkx_labels(G, pos, labels,
                              font_size=8,
                              font_weight='bold',
                              bbox=dict(facecolor='white',
                                      alpha=0.7,
                                      edgecolor='none',
                                      pad=0.5))
        
        plt.title(title, pad=20)
        plt.legend(loc='upper right')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def save_graph_data(self, G: nx.Graph, output_dir: Path, graph_type: str):
        """保存图数据为CSV和可视化"""
        # 保存节点数据
        nodes_data = []
        for node, attrs in G.nodes(data=True):
            node_data = {'node_id': node}
            node_data.update(attrs)
            nodes_data.append(node_data)
        
        pd.DataFrame(nodes_data).to_csv(
            output_dir / f'{graph_type}_nodes.csv', 
            index=False
        )
        
        # 保存边数据
        edges_data = []
        for u, v, attrs in G.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                'relationship': attrs.get('relationship', ''),
                'relation': attrs.get('relation', '')
            }
            edges_data.append(edge_data)
        
        pd.DataFrame(edges_data).to_csv(
            output_dir / f'{graph_type}_edges.csv', 
            index=False
        )
        
        # 保存GraphML格式
        nx.write_graphml(G, output_dir / f'{graph_type}_graph.graphml')

    def process_patient_data(self, story_path: Path, routine_path: Path):
        """处理单个病人的数据"""
        # 处理生平故事
        story_result = self.create_story_graph(story_path)
        if story_result:
            story_graph, patient_id = story_result
            patient_story_dir = self.story_graphs_dir / patient_id
            patient_story_dir.mkdir(exist_ok=True)
            
            self.save_graph_data(story_graph, patient_story_dir, 'story')
            self.visualize_graph(story_graph, 
                               patient_story_dir / 'story_graph.png',
                               f"Life Story Graph - {patient_id}")
        
        # 处理日常活动
        routine_result = self.create_routine_graph(routine_path)
        if routine_result:
            routine_graph, patient_id = routine_result
            patient_routine_dir = self.routine_graphs_dir / patient_id
            patient_routine_dir.mkdir(exist_ok=True)
            
            self.save_graph_data(routine_graph, patient_routine_dir, 'routine')
            self.visualize_graph(routine_graph,
                               patient_routine_dir / 'routine_graph.png',
                               f"Daily Routine Graph - {patient_id}")

    def process_all_patients(self, data_dir: Path = Path("Patient_Data")):
        """处理所有病人数据"""
        stories_dir = data_dir / "Life_Stories"
        routines_dir = data_dir / "Daily_Routines"
        
        processed_count = 0
        failed_count = 0
        
        # 获取所有病人ID
        patient_ids = {f.stem.split('_')[0] for f in stories_dir.glob("P*.json")}
        
        for patient_id in sorted(patient_ids):
            print(f"Processing {patient_id}...")
            
            story_file = stories_dir / f"{patient_id}_story.json"
            routine_file = routines_dir / f"{patient_id}_routine.json"
            
            if story_file.exists() and routine_file.exists():
                try:
                    self.process_patient_data(story_file, routine_file)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {patient_id}: {e}")
                    failed_count += 1
            else:
                print(f"Missing files for {patient_id}")
                failed_count += 1
        
        # 创建处理摘要
        self.create_graphs_summary()
        
        print(f"\nProcessing completed:")
        print(f"- Successfully processed: {processed_count}")
        print(f"- Failed: {failed_count}")
        print(f"- Graphs saved in:")
        print(f"  - Life Story Graphs: {self.story_graphs_dir}")
        print(f"  - Daily Routine Graphs: {self.routine_graphs_dir}")

    def create_graphs_summary(self):
        """创建所有图谱的摘要信息"""
        summary_data = []
        
        # 处理生平故事图谱
        for patient_dir in self.story_graphs_dir.iterdir():
            if patient_dir.is_dir():
                nodes_file = patient_dir / 'story_nodes.csv'
                edges_file = patient_dir / 'story_edges.csv'
                
                if nodes_file.exists() and edges_file.exists():
                    nodes_df = pd.read_csv(nodes_file)
                    edges_df = pd.read_csv(edges_file)
                    
                    summary_data.append({
                        'patient_id': patient_dir.name,
                        'graph_type': 'life_story',
                        'total_nodes': len(nodes_df),
                        'person_nodes': len(nodes_df[nodes_df['node_type'] == 'person']),
                        'event_nodes': len(nodes_df[nodes_df['node_type'] == 'event']),
                        'total_edges': len(edges_df)
                    })
        
        # 处理日常活动图谱
        for patient_dir in self.routine_graphs_dir.iterdir():
            if patient_dir.is_dir():
                nodes_file = patient_dir / 'routine_nodes.csv'
                edges_file = patient_dir / 'routine_edges.csv'
                
                if nodes_file.exists() and edges_file.exists():
                    nodes_df = pd.read_csv(nodes_file)
                    edges_df = pd.read_csv(edges_file)
                    
                    summary_data.append({
                        'patient_id': patient_dir.name,
                        'graph_type': 'daily_routine',
                        'total_nodes': len(nodes_df),
                        'person_nodes': len(nodes_df[nodes_df['node_type'] == 'person']),
                        'activity_nodes': len(nodes_df[nodes_df['node_type'] == 'activity']),
                        'total_edges': len(edges_df)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.base_dir / 'graphs_summary.csv', index=False)
            


def main():
    processor = SeparatedKnowledgeGraphs()
    processor.process_all_patients()

if __name__ == "__main__":
    main()