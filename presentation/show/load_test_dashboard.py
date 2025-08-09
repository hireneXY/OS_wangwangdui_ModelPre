import streamlit as st
import time
import psutil
import torch
import pandas as pd
import os
import json
from workloads import MemoryWorkload, IOWorkload
from kernel_utils import KernelTuner

# 设置页面配置
st.set_page_config(page_title="智能体内核调参展示平台", layout="wide")

class LoadTestingDashboard:
    def __init__(self):
        self.memory_workload = MemoryWorkload()
        self.io_workload = IOWorkload()
        self.kernel_tuner = KernelTuner()
        self.agent = self._load_agent()
        self.test_history = self._load_history()
        
        # 定义参数范围
        self.parameter_ranges = {
            "内存密集型": {
                "easy": {"memory_size_mb": (50, 200), "operations": (1000, 10000)},
                "medium": {"memory_size_mb": (200, 500), "operations": (5000, 20000)},
                "hard": {"memory_size_mb": (500, 800), "operations": (10000, 30000)},
                "extreme": {"memory_size_mb": (800, 1200), "operations": (15000, 50000)}
            },
            "IO密集型": {
                "easy": {"num_files": (2, 8), "file_size_mb": (5, 25), "read_operations": (10, 100)},
                "medium": {"num_files": (8, 15), "file_size_mb": (25, 60), "read_operations": (20, 150)},
                "hard": {"num_files": (15, 25), "file_size_mb": (60, 100), "read_operations": (50, 200)},
                "extreme": {"num_files": (25, 40), "file_size_mb": (100, 150), "read_operations": (100, 300)}
            },
            "文件缓存负载": {
                "easy": {"num_files": (200, 800), "file_size_kb": (10, 60), "access_operations": (300, 1000)},
                "medium": {"num_files": (800, 1500), "file_size_kb": (60, 120), "access_operations": (500, 1500)},
                "hard": {"num_files": (1500, 2500), "file_size_kb": (120, 200), "access_operations": (800, 2000)},
                "extreme": {"num_files": (2500, 4000), "file_size_kb": (200, 300), "access_operations": (1200, 2500)}
            }
        }

    def _load_agent(self):
        """加载调参智能体模型"""
        try:
            model_path = "models/ppo_memory_opt_final_20250620_001016.pt.pt"
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                class SimplePPOAgent(torch.nn.Module):
                    def __init__(self):
                        super(SimplePPOAgent, self).__init__()
                        # 根据实际模型结构调整网络
                        self.feature_extractor = torch.nn.Sequential(
                            torch.nn.Linear(21, 256),  # 21个输入特征
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, 256),
                            torch.nn.ReLU()
                        )
                        self.actor_mean = torch.nn.Linear(256, 6)  # 动作均值，6个输出
                        self.action_log_std = torch.nn.Parameter(torch.zeros(1, 6))  # 动作标准差，6个输出
                        # 添加critic网络（虽然我们不需要使用它）
                        self.critic_network = torch.nn.Sequential(
                            torch.nn.Linear(256, 128),  # 256 -> 128
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1)     # 128 -> 1
                        )
                        
                    def forward(self, x):
                        features = self.feature_extractor(x)
                        action_mean = self.actor_mean(features)
                        # 使用sigmoid确保输出在[0,1]范围内
                        actions = torch.sigmoid(action_mean)
                        return actions
                
                agent = SimplePPOAgent()
                # 加载模型权重
                agent.load_state_dict(checkpoint['model'])
                agent.eval()
                st.success("✅ 模型加载成功")
                return agent
            else:
                st.warning("⚠️ 模型文件格式不正确")
                return None
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            st.error(f"错误详情: {type(e).__name__}: {str(e)}")
            return None

    def get_system_state(self):
        """获取当前系统状态"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        }

    def generate_random_loads(self):
        """生成随机连续负载"""
        import random
        num_loads = random.randint(3, 6)
        random_loads = []
        
        for i in range(num_loads):
            load_type = random.choice(["内存密集型", "IO密集型", "文件缓存负载"])
            difficulty = random.choice(["easy", "medium", "hard", "extreme"])
            ranges = self.parameter_ranges[load_type][difficulty]
            
            load_config = {
                "id": i,
                "type": load_type,
                "difficulty": difficulty,
                "display_name": f"{difficulty}{load_type}负载"
            }
            
            if load_type == "内存密集型":
                load_config.update({
                    "memory_size_mb": random.randint(*ranges["memory_size_mb"]),
                    "operations": random.randint(*ranges["operations"])
                })
            elif load_type == "IO密集型":
                load_config.update({
                    "num_files": random.randint(*ranges["num_files"]),
                    "file_size_mb": random.randint(*ranges["file_size_mb"]),
                    "read_operations": random.randint(*ranges["read_operations"])
                })
            else:
                load_config.update({
                    "num_files": random.randint(*ranges["num_files"]),
                    "file_size_kb": random.randint(*ranges["file_size_kb"]),
                    "access_operations": random.randint(*ranges["access_operations"])
                })
            
            random_loads.append(load_config)
        
        return random_loads

    def run_custom_loads(self, custom_loads):
        """执行自定义负载配置"""
        results = []
        initial_state = self.get_system_state()
        
        # 检查权限状态
        username, uid = self.kernel_tuner.get_current_user_info()
        is_root = self.kernel_tuner.check_root_permission()
        
        # 显示权限状态
        if is_root:
            st.success(f"✅ 当前用户: {username} (UID: {uid}) - 具有root权限")
        else:
            st.warning(f"⚠️ 当前用户: {username} (UID: {uid}) - 无root权限")
            st.info("💡 建议使用 'sudo streamlit run load_test_dashboard.py' 启动应用以获得完整功能")
        
        # 获取调参前参数
        original_params = {}
        for param in ["vm.swappiness", "vm.dirty_ratio", "net.ipv4.tcp_rmem"]:
            original_params[param] = self.kernel_tuner.get_kernel_param(param)
        
        # 调用智能体进行调参
        param_values = {}
        apply_result = False
        if self.agent:
            try:
                # 确保所有键都存在
                cpu_percent = initial_state.get("cpu_percent", 0.0)
                memory_percent = initial_state.get("memory_percent", 0.0)
                disk_usage_percent = initial_state.get("disk_usage_percent", 0.0)
                load_average = initial_state.get("load_average", 0.0)
                
                # 构建21维状态向量（根据实际模型要求）
                # 前5个是系统状态，其余用0填充
                state_vector = torch.zeros(1, 21, dtype=torch.float32)
                state_vector[0, 0] = cpu_percent / 100.0
                state_vector[0, 1] = memory_percent / 100.0
                state_vector[0, 2] = disk_usage_percent / 100.0
                state_vector[0, 3] = load_average / 10.0
                state_vector[0, 4] = len(custom_loads) / 10.0
                # 其余16个维度保持为0（模型训练时可能使用的其他特征）
                    
                with torch.no_grad():
                    model_output = self.agent(state_vector)
                
                param_values = self._parse_tuning_params(model_output)
                
                # 显示调参方案
                st.info("🤖 智能体调参方案:")
                for param, value in param_values.items():
                    st.write(f"  - {param}: {value}")
                
                # 应用调参
                if is_root:
                    apply_result = True
                    success_count = 0
                    total_count = len(param_values)
                    
                    for param, value in param_values.items():
                        if self.kernel_tuner.set_kernel_param(param, value):
                            success_count += 1
                        else:
                            apply_result = False
                    
                    if success_count == total_count:
                        st.success(f"✅ 所有参数调整成功 ({success_count}/{total_count})")
                    else:
                        st.warning(f"⚠️ 部分参数调整失败 ({success_count}/{total_count})")
                else:
                    st.warning("⚠️ 由于权限不足，跳过参数调整")
                    apply_result = False
                    
            except Exception as e:
                st.error(f"智能体调参失败: {e}")
                st.error(f"错误详情: {type(e).__name__}: {str(e)}")
        
        # 执行每个负载
        for load_config in custom_loads:
            try:
                if load_config["type"] == "内存密集型":
                    runtime, memory_used = self.memory_workload.run_memory_intensive_custom(
                        load_config["memory_size_mb"], 
                        load_config["operations"]
                    )
                    final_state = self.get_system_state()
                    performance = {
                        "运行时间(秒)": round(runtime, 3),
                        "内存使用(MB)": memory_used,
                        "最终内存使用率(%)": round(final_state["memory_percent"], 1)
                    }
                elif load_config["type"] == "IO密集型":
                    runtime, num_files, read_mb, write_mb = self.io_workload.run_io_intensive_custom(
                        load_config["num_files"],
                        load_config["file_size_mb"],
                        load_config["read_operations"]
                    )
                    performance = {
                        "运行时间(秒)": round(runtime, 3),
                        "文件数量": num_files,
                        "读取数据(MB)": round(read_mb, 1),
                        "写入数据(MB)": round(write_mb, 1)
                    }
                else:  # 文件缓存负载
                    runtime, num_files, file_size_kb = self.io_workload.run_file_cache_intensive_custom(
                        load_config["num_files"],
                        load_config["file_size_kb"],
                        load_config["access_operations"]
                    )
                    performance = {
                        "运行时间(秒)": round(runtime, 3),
                        "文件数量": num_files,
                        "平均文件大小(KB)": file_size_kb
                    }
            
                results.append({
                    "负载名称": load_config["display_name"],
                    "调参方案": param_values,
                    "调参前参数": original_params,
                    "性能指标": performance,
                    "调参成功": apply_result
                })
                
            except Exception as e:
                st.error(f"执行负载 {load_config['display_name']} 失败: {e}")
                results.append({
                    "负载名称": load_config["display_name"],
                    "调参方案": param_values,
                    "调参前参数": original_params,
                    "性能指标": {"运行时间(秒)": 0, "错误": str(e)},
                    "调参成功": apply_result
                })
        
        # 保存到历史记录
        history_entry = {
            "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "负载组合": [load["display_name"] for load in custom_loads],
            "结果": results
        }
        self.test_history.append(history_entry)
        self._save_history()
        
        return results

    def _parse_tuning_params(self, model_output):
        """解析模型输出为实际内核参数值"""
        # 模型输出6个值，我们只使用前3个来设置3个内核参数
        return {
            "vm.swappiness": int(model_output[0][0].item() * 100),
            "vm.dirty_ratio": int(model_output[0][1].item() * 45) + 5,
            "net.ipv4.tcp_rmem": f"{int(model_output[0][2].item()*2000) + 4096} 87380 6291456"
        }

    def _save_history(self):
        """保存历史记录到文件"""
        try:
            with open("test_history.json", 'w', encoding='utf-8') as f:
                json.dump(self.test_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"保存历史记录失败: {e}")

    def _load_history(self):
        """从文件加载历史记录"""
        try:
            if os.path.exists("test_history.json"):
                with open("test_history.json", 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.warning(f"加载历史记录失败: {e}")
            return []

    def show_history(self):
        """展示历史测试记录"""
        if not self.test_history:
            st.info("暂无测试历史")
            return
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("测试历史记录")
        with col2:
            if st.button("清空历史记录", type="secondary"):
                self.test_history = []
                self._save_history()
                st.rerun()
        
        st.info(f"共保存了 {len(self.test_history)} 次测试记录")
        
        for i, history in enumerate(reversed(self.test_history)):
            with st.expander(f"测试 {len(self.test_history) - i}: {history['时间']} - 负载: {', '.join(history['负载组合'])}", expanded=False):
                for j, result in enumerate(history["结果"]):
                    st.markdown(f"### {j+1}. {result['负载名称']}")
                    
                    if result['调参方案']:
                        st.markdown("#### 智能体调参方案")
                        comparison_data = []
                        for param, new_value in result['调参方案'].items():
                            original_value = result.get('调参前参数', {}).get(param, "未知")
                            comparison_data.append({
                                "参数": param,
                                "调参前": original_value,
                                "调参后": new_value,
                                "变化": f"{original_value} → {new_value}"
                            })
                        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                        
                        if result.get('调参成功'):
                            st.success("✅ 内核参数调整成功")
                        else:
                            st.warning("⚠️ 部分参数调整未生效（可能需要root权限）")
                    
                    st.markdown("#### 性能指标")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("运行时间", f"{result['性能指标']['运行时间(秒)']}秒")
                    with col2:
                        if "内存使用(MB)" in result['性能指标']:
                            st.metric("内存消耗", f"{result['性能指标']['内存使用(MB)']}MB")
                        elif "文件数量" in result['性能指标']:
                            st.metric("文件数量", f"{result['性能指标']['文件数量']}")
                    with col3:
                        if "最终内存使用率(%)" in result['性能指标']:
                            st.metric("系统内存使用率", f"{result['性能指标']['最终内存使用率(%)']}%")
                        elif "读取数据(MB)" in result['性能指标']:
                            st.metric("读取数据", f"{result['性能指标']['读取数据(MB)']}MB")
                    
                    st.write("---")

def main():
    st.title("智能体内核调参展示平台")
    dashboard = LoadTestingDashboard()
    
    # 左侧：负载选择区域
    with st.sidebar:
        st.header("负载配置")
        
        # 初始化session_state
        if 'custom_loads' not in st.session_state:
            st.session_state.custom_loads = []
        if 'load_counter' not in st.session_state:
            st.session_state.load_counter = 0
        if 'delete_confirm' not in st.session_state:
            st.session_state.delete_confirm = None
        
        # 快速选项
        st.subheader("快速选项")
        if st.button("🎲 随机连续负载", type="secondary"):
            st.session_state.custom_loads = dashboard.generate_random_loads()
            st.rerun()
        
        if st.button("🗑️ 清空所有负载", type="secondary"):
            if st.session_state.custom_loads:
                st.session_state.custom_loads = []
                st.session_state.delete_confirm = None
                st.success("已清空所有负载")
                st.rerun()
            else:
                st.info("当前没有配置的负载")
        
        st.write("---")
        
        # 自定义负载配置
        st.subheader("自定义负载配置")
        
        load_type = st.selectbox("选择负载类型", ["内存密集型", "IO密集型", "文件缓存负载"], key="load_type")
        difficulty = st.selectbox("选择难度", ["easy", "medium", "hard", "extreme"], key="difficulty")
        
        # 根据负载类型和难度显示参数配置
        if load_type == "内存密集型":
            st.markdown("**内存配置**")
            if difficulty == "easy":
                memory_size = st.slider("内存大小 (MB)", 50, 200, 100, key="memory_size")
            elif difficulty == "medium":
                memory_size = st.slider("内存大小 (MB)", 200, 500, 300, key="memory_size")
            elif difficulty == "hard":
                memory_size = st.slider("内存大小 (MB)", 500, 800, 600, key="memory_size")
            else:  # extreme
                memory_size = st.slider("内存大小 (MB)", 800, 1200, 1000, key="memory_size")
            operations = st.slider("操作次数", 1000, 50000, 10000, key="operations")
            
        elif load_type == "IO密集型":
            st.markdown("**IO配置**")
            if difficulty == "easy":
                num_files = st.slider("文件数量", 2, 8, 5, key="num_files")
                file_size = st.slider("文件大小 (MB)", 5, 25, 15, key="file_size")
            elif difficulty == "medium":
                num_files = st.slider("文件数量", 8, 15, 12, key="num_files")
                file_size = st.slider("文件大小 (MB)", 25, 60, 40, key="file_size")
            elif difficulty == "hard":
                num_files = st.slider("文件数量", 15, 25, 20, key="num_files")
                file_size = st.slider("文件大小 (MB)", 60, 100, 80, key="file_size")
            else:  # extreme
                num_files = st.slider("文件数量", 25, 40, 30, key="num_files")
                file_size = st.slider("文件大小 (MB)", 100, 150, 120, key="file_size")
            read_operations = st.slider("读取操作次数", 10, 200, 50, key="read_operations")
            
        else:  # 文件缓存负载
            st.markdown("**缓存配置**")
            if difficulty == "easy":
                num_files = st.slider("文件数量", 200, 800, 500, key="cache_files")
                file_size = st.slider("文件大小 (KB)", 10, 60, 30, key="cache_size")
            elif difficulty == "medium":
                num_files = st.slider("文件数量", 800, 1500, 1200, key="cache_files")
                file_size = st.slider("文件大小 (KB)", 60, 120, 90, key="cache_size")
            elif difficulty == "hard":
                num_files = st.slider("文件数量", 1500, 2500, 2000, key="cache_files")
                file_size = st.slider("文件大小 (KB)", 120, 200, 160, key="cache_size")
            else:  # extreme
                num_files = st.slider("文件数量", 2500, 4000, 3000, key="cache_files")
                file_size = st.slider("文件大小 (KB)", 200, 300, 250, key="cache_size")
            access_operations = st.slider("访问操作次数", 300, 2500, 1000, key="access_operations")
        
        # 添加负载按钮
        if st.button("添加此负载", type="primary"):
            load_config = {
                "id": st.session_state.load_counter,
                "type": load_type,
                "difficulty": difficulty,
                "display_name": f"{difficulty}{load_type}负载"
            }
            
            # 添加具体参数
            if load_type == "内存密集型":
                load_config.update({"memory_size_mb": memory_size, "operations": operations})
            elif load_type == "IO密集型":
                load_config.update({"num_files": num_files, "file_size_mb": file_size, "read_operations": read_operations})
            else:  # 文件缓存负载
                load_config.update({"num_files": num_files, "file_size_kb": file_size, "access_operations": access_operations})
            
            st.session_state.custom_loads.append(load_config)
            st.session_state.load_counter += 1
            st.rerun()
        
        st.write("---")
        
        # 显示已配置的负载
        if st.session_state.custom_loads:
            st.subheader(f"已配置的负载 ({len(st.session_state.custom_loads)} 个)")
            for i, load in enumerate(st.session_state.custom_loads):
                with st.expander(f"{i+1}. {load['display_name']}", expanded=False):
                    # 负载详情
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**类型**: {load['type']}")
                        st.write(f"**难度**: {load['difficulty']}")
                        
                        if load['type'] == "内存密集型":
                            st.write(f"**内存大小**: {load['memory_size_mb']}MB")
                            st.write(f"**操作次数**: {load['operations']}")
                        elif load['type'] == "IO密集型":
                            st.write(f"**文件数量**: {load['num_files']}")
                            st.write(f"**文件大小**: {load['file_size_mb']}MB")
                            st.write(f"**读取操作**: {load['read_operations']}")
                        else:  # 文件缓存负载
                            st.write(f"**文件数量**: {load['num_files']}")
                            st.write(f"**文件大小**: {load['file_size_kb']}KB")
                            st.write(f"**访问操作**: {load['access_operations']}")
                    
                    # 操作按钮
                    with col2:
                        # 删除按钮
                        if st.button("🗑️ 删除", key=f"delete_load_{i}_{load['id']}", type="secondary"):
                            # 设置确认删除状态
                            st.session_state.delete_confirm = i
                            st.rerun()
                        
                        # 复制按钮
                        if st.button("📋 复制", key=f"copy_load_{i}_{load['id']}", type="secondary"):
                            # 复制负载配置
                            copied_load = load.copy()
                            copied_load['id'] = st.session_state.load_counter
                            copied_load['display_name'] = f"{copied_load['display_name']}_副本"
                            st.session_state.custom_loads.append(copied_load)
                            st.session_state.load_counter += 1
                            st.success(f"已复制负载: {load['display_name']}")
                            st.rerun()
                        
                        # 确认删除对话框
                        if st.session_state.delete_confirm == i:
                            st.warning(f"确认删除负载: {load['display_name']}?")
                            col_confirm1, col_confirm2 = st.columns(2)
                            with col_confirm1:
                                if st.button("✅ 确认", key=f"confirm_delete_{i}"):
                                    # 从列表中移除该负载
                                    deleted_load = st.session_state.custom_loads.pop(i)
                                    st.session_state.delete_confirm = None
                                    st.success(f"已删除负载: {deleted_load['display_name']}")
                                    st.rerun()
                            with col_confirm2:
                                if st.button("❌ 取消", key=f"cancel_delete_{i}"):
                                    st.session_state.delete_confirm = None
                                    st.rerun()
        
        # 执行按钮
        if st.session_state.custom_loads:
            if st.button("开始执行配置的负载", type="primary"):
                results = dashboard.run_custom_loads(st.session_state.custom_loads)
                st.session_state['last_results'] = results
        else:
            st.info("请先配置至少一个负载")
        
        # 重置参数按钮
        if st.button("重置内核参数为默认值"):
            result = dashboard.kernel_tuner.reset_default_params()
            if result:
                st.success("内核参数已重置为默认值")
            else:
                st.warning("部分参数重置失败（可能需要root权限）")
    
    # 右侧：结果展示区域
    tab1, tab2 = st.tabs(["实时结果", "历史记录"])
    
    with tab1:
        if 'last_results' in st.session_state and st.session_state['last_results']:
            st.subheader("最新测试结果")
            results = st.session_state['last_results']
            
            for result in results:
                with st.expander(f"**{result['负载名称']}**", expanded=True):
                    if result['调参方案']:
                        st.subheader("智能体调参方案")
                        comparison_data = []
                        for param, new_value in result['调参方案'].items():
                            original_value = result.get('调参前参数', {}).get(param, "未知")
                            comparison_data.append({
                                "参数": param,
                                "调参前": original_value,
                                "调参后": new_value,
                                "变化": f"{original_value} → {new_value}"
                            })
                        st.dataframe(pd.DataFrame(comparison_data))
                        
                        if result.get('调参成功'):
                            st.success("✅ 内核参数调整成功")
                        else:
                            st.warning("⚠️ 部分参数调整未生效（可能需要root权限）")
                    
                    st.subheader("性能指标")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("运行时间", f"{result['性能指标']['运行时间(秒)']}秒")
                    with col2:
                        if "内存使用(MB)" in result['性能指标']:
                            st.metric("内存消耗", f"{result['性能指标']['内存使用(MB)']}MB")
                        elif "文件数量" in result['性能指标']:
                            st.metric("文件数量", f"{result['性能指标']['文件数量']}")
                    with col3:
                        if "最终内存使用率(%)" in result['性能指标']:
                            st.metric("系统内存使用率", f"{result['性能指标']['最终内存使用率(%)']}%")
                        elif "读取数据(MB)" in result['性能指标']:
                            st.metric("读取数据", f"{result['性能指标']['读取数据(MB)']}MB")
        else:
            st.info("请在左侧选择负载类型并点击执行按钮开始测试")
    
    with tab2:
        dashboard.show_history()

if __name__ == "__main__":
    main()
