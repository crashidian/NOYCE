from pyvis.network import Network
import pandas as pd
from pathlib import Path
import webbrowser

def create_stable_graph(nodes_df, edges_df, output_path):
    """创建带有图例的稳定网络图"""
    net = Network(
        height="900px",
        width="100%", 
        bgcolor="#f8f9fa",
        font_color="black",
        directed=False
    )
    
    # 使用更稳定的物理引擎设置
    net.barnes_hut(
        gravity=-2000,          
        central_gravity=0.2,     
        spring_length=150,
        spring_strength=0.01,    
        damping=0.95,           
        overlap=0.01
    )
    
    # 节点样式定义
    node_styles = {
        'person': {
            'patient': {
                'color': '#e74c3c',
                'shape': 'dot', 
                'size': 50,
                'font_size': 36
            },
            'caregiver': {'color': '#3498db', 'shape': 'dot', 'size': 35, 'font_size': 36},
            'family': {'color': '#2ecc71', 'shape': 'dot', 'size': 35, 'font_size': 36},
            'staff': {'color': '#9b59b6', 'shape': 'dot', 'size': 35, 'font_size': 36},
            'other': {'color': '#f1c40f', 'shape': 'dot', 'size': 35, 'font_size': 36}
        },
        'activity': {
            'medical': {'color': '#e67e22', 'shape': 'square', 'size': 15, 'font_size': 36},
            'daily': {'color': '#16a085', 'shape': 'square', 'size': 15, 'font_size': 36},
            'social': {'color': '#34495e', 'shape': 'square', 'size': 15, 'font_size': 36},
            'therapy': {'color': '#8e44ad', 'shape': 'square', 'size': 15, 'font_size': 36},
            'default': {'color': '#7f8c8d', 'shape': 'square', 'size': 15, 'font_size': 36}
        }
    }
    
    # 添加节点
    for _, row in nodes_df.iterrows():
        node_type = row['node_type']
        
        # 确定节点样式
        if node_type == 'person':
            role = row.get('role', '').lower()
            # 检查是否是主要病人节点
            if 'patient' in role and any(pid in row['node_id'] for pid in ['P001', 'P002', 'P003', 'P004', 'P005']):
                style = node_styles['person']['patient']
            elif 'caregiver' in role:
                style = node_styles['person']['caregiver']
            elif 'family' in role:
                style = node_styles['person']['family']
            elif any(x in role for x in ['nurse', 'doctor', 'therapist']):
                style = node_styles['person']['staff']
            else:
                style = node_styles['person']['other']
        else:
            activity_type = 'default'
            activity_name = row.get('activity_type', '').lower()
            if any(x in activity_name for x in ['medication', 'medical', 'therapy']):
                activity_type = 'medical'
            elif any(x in activity_name for x in ['breakfast', 'lunch', 'dinner', 'routine']):
                activity_type = 'daily'
            elif any(x in activity_name for x in ['group', 'family', 'interaction']):
                activity_type = 'social'
            elif any(x in activity_name for x in ['therapy', 'workshop']):
                activity_type = 'therapy'
            style = node_styles['activity'][activity_type]
        
        # 创建悬停提示
        title = f"<div style='font-size:20px; padding:15px;'>"
        title += f"<b>{row.get('display_name', '')}</b><br><br>"
        for key, value in row.items():
            if key not in ['node_id', 'node_type', 'display_name']:
                title += f"{key}: {value}<br>"
        title += "</div>"
        
        # 添加节点
        net.add_node(
            row['node_id'],
            label=row['display_name'],
            title=title,
            color=style['color'],
            shape=style['shape'],
            size=style['size'],
            borderWidth=3,
            font={
                'size': style['font_size'],
                'face': 'arial',
                'color': 'black',
                'strokeWidth': 5,
                'strokeColor': '#ffffff'
            }
        )
    
    # 添加边
    for _, row in edges_df.iterrows():
        net.add_edge(
            row['source'],
            row['target'],
            title=row.get('relation', '') or row.get('relationship', ''),
            color={'color': '#2d3436', 'opacity': 0.5},
            width=2,
            smooth={'type': 'continuous', 'roundness': 0.2}
        )
    
    # 设置高级选项
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.95,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75,
        "maxVelocity": 30,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 2000,
          "updateInterval": 50,
          "onlyDynamicEdges": false,
          "fit": true
        },
        "timestep": 0.3,
        "adaptiveTimestep": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "nodes": {
        "font": {
          "strokeWidth": 36,
          "strokeColor": "#ffffff"
        },
        "scaling": {
          "label": {
            "enabled": false
          }
        },
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.3)",
          "size": 12,
          "x": 5,
          "y": 5
        }
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "continuous",
          "roundness": 0.2
        },
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.2)",
          "size": 5,
          "x": 3,
          "y": 3
        }
      }
    }
    """)
    
    # 添加CSS样式和图例
    net.html += """
    <style>
    body {
        background-color: #f8f9fa;
        margin: 0;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    
    #mynetwork {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .legend-container {
        position: absolute;
        top: 20px;
        right: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1000;
        max-width: 300px;
    }
    
    .legend-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 2px solid #ddd;
        padding-bottom: 5px;
    }
    
    .legend-section {
        margin-bottom: 15px;
    }
    
    .legend-section-title {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
        color: #2c3e50;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        font-size: 12px;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        margin-right: 8px;
        border-radius: 3px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .legend-circle {
        border-radius: 50%;
    }
    
    .legend-square {
        border-radius: 3px;
    }
    </style>
    
    <div class="legend-container">
        <div class="legend-title">图例说明</div>
        
        <!-- 人物节点图例 -->
        <div class="legend-section">
            <div class="legend-section-title">人物节点</div>
            <div class="legend-item">
                <div class="legend-color legend-circle" style="background-color: #e74c3c;"></div>
                <span>病人（较大圆形）</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-circle" style="background-color: #3498db;"></div>
                <span>照护者</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-circle" style="background-color: #2ecc71;"></div>
                <span>家庭成员</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-circle" style="background-color: #9b59b6;"></div>
                <span>医护人员</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-circle" style="background-color: #f1c40f;"></div>
                <span>其他人员</span>
            </div>
        </div>
        
        <!-- 活动节点图例 -->
        <div class="legend-section">
            <div class="legend-section-title">活动节点（方形）</div>
            <div class="legend-item">
                <div class="legend-color legend-square" style="background-color: #e67e22;"></div>
                <span>医疗活动</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-square" style="background-color: #16a085;"></div>
                <span>日常活动</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-square" style="background-color: #34495e;"></div>
                <span>社交活动</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-square" style="background-color: #8e44ad;"></div>
                <span>治疗活动</span>
            </div>
        </div>
        
        <!-- 使用说明 -->
        <div class="legend-section">
            <div class="legend-section-title">操作说明</div>
            <div style="font-size: 12px; color: #666;">
                • 拖动节点可调整位置<br>
                • 滚轮缩放查看详情<br>
                • 悬停节点显示信息<br>
                • 可拖动整个视图
            </div>
        </div>
    </div>
    
    <script>
    // 监听网络稳定化事件
    network.once("stabilizationIterationsDone", function() {
        // 稳定后的操作
        document.querySelectorAll('.legend-container').forEach(function(el) {
            el.style.opacity = '1';
        });
    });
    </script>
    """
    
    try:
        net.write_html(output_path)
        return True
    except Exception as e:
        print(f"Error saving network: {e}")
        return False

def visualize_patient_data_interactive(source_folder: str, target_folder: str, num_graphs: int = 5):
    """为病人数据创建交互式可视化"""
    source_path = Path(source_folder).resolve()
    output_dir = Path(target_folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted([d for d in source_path.glob('P*') 
                         if d.is_dir()])[:num_graphs]
    
    print(f"\nFound {len(patient_dirs)} patient directories")
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        print(f"Processing {patient_id}...")
        
        nodes_file = patient_dir / 'routine_nodes.csv'
        edges_file = patient_dir / 'routine_edges.csv'
        
        if nodes_file.exists() and edges_file.exists():
            try:
                nodes_df = pd.read_csv(nodes_file)
                edges_df = pd.read_csv(edges_file)
                
                output_file = output_dir / f"{patient_id}_interactive.html"
                
                if create_stable_graph(nodes_df, edges_df, str(output_file)):
                    print(f"Created visualization: {output_file}")
                    webbrowser.open(f'file://{output_file.absolute()}')
                    
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
        else:
            print(f"Missing required files for {patient_id}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    source_folder = "Knowledge_Graphs/Daily_Routine_Graphs"
    target_folder = "Interactive_Visualizations"
    
    visualize_patient_data_interactive(source_folder, target_folder, num_graphs=3)