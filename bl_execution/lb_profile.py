import time
import threading
import psutil, GPUtil

from contextlib import ContextDecorator
from collections import defaultdict
from typing import Dict, List, Optional


# 全局状态管理 - 使用 threading.local()
_profiler_state = threading.local()

def _init_profiler_state():
    """初始化线程本地状态"""
    if not hasattr(_profiler_state, 'active_profiler'):
        _profiler_state.active_profiler = None
    if not hasattr(_profiler_state, 'current_node'):
        _profiler_state.current_node = None
    if not hasattr(_profiler_state, 'node_stack'):
        _profiler_state.node_stack = []


class NodeRecord:
    """节点性能记录"""
    
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.children = defaultdict(lambda: NodeRecord(""))
        
    def add_time(self, elapsed: float):
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        
    def add_resource_usage(self, cpu: float, memory: float, gpu_memory: float = 0.0):
        if cpu >= 0:
            self.cpu_usage.append(cpu)
        if memory >= 0:
            self.memory_usage.append(memory)
        if gpu_memory >= 0:
            self.gpu_memory_usage.append(gpu_memory)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        avg_time = self.total_time / max(self.call_count, 1)
        avg_cpu = sum(self.cpu_usage) / max(len(self.cpu_usage), 1) if self.cpu_usage else 0.0
        avg_memory = sum(self.memory_usage) / max(len(self.memory_usage), 1) if self.memory_usage else 0.0
        avg_gpu = sum(self.gpu_memory_usage) / max(len(self.gpu_memory_usage), 1) if self.gpu_memory_usage else 0.0
        
        return {
            'name': self.name,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'avg_gpu_memory': avg_gpu,
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        stats = self.get_statistics()
        stats['children'] = {k: v.to_dict() for k, v in self.children.items()}
        return stats
    

class record_func(ContextDecorator):
    """标记函数，用于统计节点性能"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.start_resources = None
        
    def _get_resources(self) -> Dict[str, float]:
        """获取当前资源使用情况"""
        process = psutil.Process()
        
        # CPU使用率
        try:
            cpu_percent = process.cpu_percent(interval=0.001)
        except:
            cpu_percent = 0.0
        
        # 内存使用
        try:
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
        except:
            memory_mb = 0.0
        
        # GPU显存使用
        gpu_memory = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = gpus[0].memoryUsed
        except:
            pass
            
        return {
            'cpu': cpu_percent,
            'memory': memory_mb,
            'gpu_memory': gpu_memory
        }
    
    def __enter__(self):
        # 确保线程本地状态已初始化
        _init_profiler_state()
        
        profiler = getattr(_profiler_state, 'active_profiler', None)
        if profiler and profiler.is_recording:
            self.start_time = time.perf_counter()
            self.start_resources = self._get_resources()
            
            # 处理节点层级关系
            _profiler_state.node_stack.append(self.name)
            _profiler_state.current_node = self.name
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 确保线程本地状态已初始化
        _init_profiler_state()
        
        profiler = getattr(_profiler_state, 'active_profiler', None)
        if profiler and profiler.is_recording and self.start_time:
            elapsed = time.perf_counter() - self.start_time
            
            # 获取结束时的资源使用
            end_resources = self._get_resources()
            
            # 计算资源使用（使用开始和结束的平均值）
            cpu_usage = (self.start_resources['cpu'] + end_resources['cpu']) / 2
            memory_usage = max(self.start_resources['memory'], end_resources['memory'])
            gpu_memory_usage = max(self.start_resources['gpu_memory'], end_resources['gpu_memory'])
            
            # 更新节点堆栈
            if _profiler_state.node_stack:
                _profiler_state.node_stack.pop()
                _profiler_state.current_node = _profiler_state.node_stack[-1] if _profiler_state.node_stack else None
            
            # 记录性能数据
            profiler.record_node(
                self.name, 
                elapsed, 
                cpu_usage, 
                memory_usage, 
                gpu_memory_usage
            )
        
        return False


class FEProfiler:
    """轻量级、非侵入式性能统计工具"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.data = defaultdict(lambda: NodeRecord(""))
        self.is_recording = False
        self.start_time = None
        self.total_record = NodeRecord("TOTAL")
        self.enable_gpu_monitoring = True
        
    def reset(self):
        """统计信息清零"""
        self.data.clear()
        self.total_record = NodeRecord("TOTAL")
        # print(f"[FEProfiler] 性能统计已重置")
    
    def start(self):
        """开始统计"""
        if self.is_recording:
            # print(f"[FEProfiler] 统计已经在进行中")
            return
            
        self.is_recording = True
        
        # 确保线程本地状态已初始化
        _init_profiler_state()
        
        # 设置活动性能分析器
        _profiler_state.active_profiler = self
        
        # 初始化栈
        _profiler_state.node_stack = []
        _profiler_state.current_node = None
        
        self.start_time = time.perf_counter()
        
        # 初始化资源基准
        self._initial_resources = self._get_current_resources()
        # print(f"[FEProfiler] 性能统计已开始")
    
    def stop(self):
        """结束统计"""
        # 确保线程本地状态已初始化
        _init_profiler_state()
        
        if not self.is_recording:
            # print(f"[FEProfiler] 统计未在进行中")
            return
            
        self.is_recording = False
        _profiler_state.active_profiler = None
        _profiler_state.current_node = None
        
        # 清空栈
        if hasattr(_profiler_state, 'node_stack'):
            _profiler_state.node_stack.clear()
        
        elapsed = time.perf_counter() - self.start_time
        self.total_record.add_time(elapsed)
        
        # print(f"[FEProfiler] 性能统计已结束，总耗时: {elapsed:.3f}秒")
    
    def record_node(self, name: str, elapsed: float, cpu_usage: float, 
                    memory_usage: float, gpu_memory_usage: float):
        """记录节点性能数据"""
        # 确保线程本地状态已初始化
        _init_profiler_state()
        
        if not self.is_recording:
            return
            
        node_record = self.data[name]
        if node_record.name == "":
            node_record.name = name
            
        node_record.add_time(elapsed)
        node_record.add_resource_usage(cpu_usage, memory_usage, gpu_memory_usage)
        
        # 如果存在父节点，也记录到父节点的children中
        if hasattr(_profiler_state, 'current_node') and _profiler_state.current_node and _profiler_state.current_node in self.data:
            parent_record = self.data[_profiler_state.current_node]
            if name not in parent_record.children:
                child_record = NodeRecord(name)
                parent_record.children[name] = child_record
            child_record = parent_record.children[name]
            child_record.add_time(elapsed)
            child_record.add_resource_usage(cpu_usage, memory_usage, gpu_memory_usage)
    
    def _get_current_resources(self) -> Dict[str, float]:
        """获取当前系统资源"""
        process = psutil.Process()
        
        resources = {
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'gpu_memory_mb': 0.0
        }
        
        try:
            resources['cpu_percent'] = process.cpu_percent(interval=0.001)
            resources['memory_mb'] = process.memory_info().rss / (1024 * 1024)
        except:
            pass
        
        if self.enable_gpu_monitoring:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    resources['gpu_memory_mb'] = gpus[0].memoryUsed
            except:
                pass
                
        return resources
    
    def display(self, output_format: str = "terminal", save_path: Optional[str] = None, 
                show_children: bool = False, sort_by: str = "total_time"):
        """展示性能信息
        
        Args:
            output_format: 输出格式，可选 "terminal", "json", "html"
            save_path: 保存路径（当output_format为json或html时使用）
            show_children: 是否显示子节点信息
            sort_by: 排序方式，可选 "name", "total_time", "avg_time", "call_count"
        """
        if not self.data:
            print("[FEProfiler] 没有性能数据可展示")
            return
        
        if output_format == "terminal":
            self._display_terminal(show_children, sort_by)
        # elif output_format == "json":
        #     self._display_json(save_path)
        # elif output_format == "html":
        #     self._display_html(save_path, show_children)
        else:
            print(f"[FEProfiler] 不支持的输出格式: {output_format}")
    
    def _get_display_data(self, show_children: bool = False) -> List[Dict]:
        """获取要显示的数据"""
        display_data = []
        
        for name, record in self.data.items():
            if record.call_count > 0:
                stats = record.get_statistics()
                
                display_data.append({
                    'name': name,
                    'avg_time': stats['avg_time'],
                    'total_time': stats['total_time'],
                    'min_time': stats['min_time'],
                    'max_time': stats['max_time'],
                    'avg_cpu': stats['avg_cpu'],
                    'avg_memory': stats['avg_memory'],
                    'avg_gpu_memory': stats['avg_gpu_memory'],
                    'call_count': stats['call_count'],
                    'is_child': False,
                    'level': 0
                })
                
                # 如果需要显示子节点
                if show_children and record.children:
                    for child_name, child_record in record.children.items():
                        if child_record.call_count > 0:
                            child_stats = child_record.get_statistics()
                            
                            display_data.append({
                                'name': f"  └─ {child_name}",
                                'avg_time': child_stats['avg_time'],
                                'total_time': child_stats['total_time'],
                                'min_time': child_stats['min_time'],
                                'max_time': child_stats['max_time'],
                                'avg_cpu': child_stats['avg_cpu'],
                                'avg_memory': child_stats['avg_memory'],
                                'avg_gpu_memory': child_stats['avg_gpu_memory'],
                                'call_count': child_stats['call_count'],
                                'is_child': True,
                                'level': 1
                            })
        
        return display_data
    
    def _display_terminal(self, show_children: bool = False, sort_by: str = "total_time"):
        """在终端显示性能数据"""
        display_data = self._get_display_data(show_children)
        
        if not display_data:
            print("[FEProfiler] 没有性能数据可显示")
            return
        
        # 排序
        reverse = sort_by in ["total_time", "avg_time", "call_count"]
        if sort_by == "name":
            display_data.sort(key=lambda x: x['name'])
        elif sort_by == "total_time":
            display_data.sort(key=lambda x: x['total_time'], reverse=reverse)
        elif sort_by == "avg_time":
            display_data.sort(key=lambda x: x['avg_time'], reverse=reverse)
        elif sort_by == "call_count":
            display_data.sort(key=lambda x: x['call_count'], reverse=reverse)
        
        # 定义列配置（名称，宽度，对齐方式）
        columns = [
            ('Node cls', 40, 'right'),
            ('Time(s)', 12, 'right'),
            ('AVG Time(s)', 12, 'right'),
            ('CPU(%)', 10, 'right'),
            ('RAM(MB)', 12, 'right'),
            ('GPU(MB)', 12, 'right'),
            ('Of Call', 10, 'right'),
        ]
        
        # 计算总宽度
        total_width = sum(col[1] for col in columns) + len(columns) + 1
        
        # 打印表头
        print("\n" + "=" * total_width)
        header = ""
        for name, width, align in columns:
            if align == 'left':
                header += f" {name:<{width}}"
            else:
                header += f" {name:>{width}}"
        header += " "
        print(header)
        print("=" * total_width)
        
        # 打印数据行
        for item in display_data:
            row = ""
            
            # 节点名称
            name = item['name']
            row += f" {name:>{columns[0][1]}}"
            
            # 总耗时
            row += f" {item['total_time']:>{columns[1][1]}.3f}"
            # 平均耗时
            row += f" {item['avg_time']:>{columns[2][1]}.3f}"
            # CPU使用率
            row += f" {item['avg_cpu']:>{columns[3][1]}.1f}"
            # 内存使用
            row += f" {item['avg_memory']:>{columns[4][1]}.1f}"
            # 显存使用
            row += f" {item['avg_gpu_memory']:>{columns[5][1]}.1f}"
            # 调用次数
            row += f" {item['call_count']:>{columns[6][1]}}"
            row += " "
            print(row)
        
        # 打印分隔线
        print("-" * total_width)
        
        # 打印总计行
        if self.total_record.call_count > 0:
            total_avg = self.total_record.total_time / self.total_record.call_count
            
            total_row = f" {'Total':>{columns[0][1]}}"
            total_row += f" {self.total_record.total_time:>{columns[1][1]}.3f}"
            total_row += f" {total_avg:>{columns[2][1]}.3f}"
            total_row += f" {'-':>{columns[3][1]}}"
            total_row += f" {'-':>{columns[4][1]}}"
            total_row += f" {'-':>{columns[5][1]}}"
            total_row += f" {self.total_record.call_count:>{columns[6][1]}}"
            total_row += " "
            
            print(total_row)
            print("=" * total_width)